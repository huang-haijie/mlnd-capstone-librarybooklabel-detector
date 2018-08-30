package org.librarybook.app;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

public class Utils {
    private static final String TAG = "Library Book Utils";
    private static final String DIR = "BookLabelDetector";

    public static void saveBitmap(Bitmap bitmap, String fileName) {
        if (bitmap == null || fileName == null || fileName.isEmpty()) {
            Log.e(TAG, "Failed to save bitmap.  Invalid parameters.");
            return;
        }

        try {
            String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + DIR;
            File rootFile = new File(rootPath);
            if (!rootFile.exists()){
                if (!rootFile.mkdirs()) {
                    Log.e(TAG, "Failed to create image root folder: " + rootPath);
                    return;
                }
            }

            File imageFile = new File(rootPath, fileName);
            if (imageFile.exists())
                imageFile.delete();

            Log.v(TAG, "Saving " + imageFile.getAbsolutePath());
            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.flush();
            fos.close();
        } catch (Exception e) {
            Log.e(TAG, "Error!", e);
        }
    }

    public static void saveBitmap(Bitmap bitmap) {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String fileName = timeStamp + "_" + String.valueOf(new Random().nextInt()) + ".jpg";
        saveBitmap(bitmap, fileName);
    }

    public static Bitmap cropBitmapFromRectF(Bitmap bitmap, RectF rectF) {
        int x = (int) rectF.left;
        int y = (int) rectF.top;
        int w = (int) rectF.width();
        int h = (int) rectF.height();

        if (x + w > bitmap.getWidth())
            w = bitmap.getWidth() - x;
        if (y + h > bitmap.getHeight())
            h = bitmap.getHeight() - y;

        return Bitmap.createBitmap(bitmap, x, y, w, h);
    }

    public static void saveText(String text, String fileName) {
        if (text == null || fileName == null || fileName.isEmpty()) {
            Log.e(TAG, "Failed to save text.  Invalid parameters.");
            return;
        }

        try {
            String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + DIR;
            File rootFile = new File(rootPath);
            if (!rootFile.exists()){
                if (!rootFile.mkdirs()) {
                    Log.e(TAG, "Failed to create text root folder: " + rootPath);
                    return;
                }
            }

            File textFile = new File(rootPath, fileName);
            FileWriter writer = new FileWriter(textFile);
            writer.write(text);
            writer.flush();
            writer.close();
        } catch (Exception e) {
            Log.e(TAG, "Error saving text!", e);
        }
    }


    public static RectF scaleRectF(RectF rect, float scaleFactor) {
        float centerX = rect.centerX();
        float centerY = rect.centerY();
        float newWidth = rect.width() * scaleFactor;
        float newHeight = rect.height() * scaleFactor;
        float newLeft = centerX - newWidth/2.0f;
        float newRight = newLeft + newWidth;
        if (newLeft<0) newLeft=0;
        float newTop = centerY - newHeight/2.0f;
        float newBottom = newTop + newHeight;
        if (newTop<0) newTop=0;
        return new RectF(newLeft, newTop, newRight, newBottom);
    }

    public static String stripString(String str) {
        if (str == null)
            return null;
        return str.replaceAll("[|.\n ]", "").toUpperCase();
    }
}
