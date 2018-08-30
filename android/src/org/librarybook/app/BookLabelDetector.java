package org.librarybook.app;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.demo.Classifier;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;

import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;
import android.widget.Toast;

import com.google.android.gms.tasks.Task;

import com.google.android.gms.tasks.Tasks;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextDetector;

public class BookLabelDetector {

    public static final int TASK_NONE = 0;
    public static final int LOCATE_BOOK = 1;
    public static final int INSERT_BOOK = 2;
    public static final int DETECT_MISPLACED =3;
    private int mDetectionTask;
    private String mDetectionText;
    private Bitmap mBitmap;
    private List<Classifier.Recognition> mRecognitions;
    private List<Pair<Integer, String>> mRecognitionTexts;
    private List<Integer> mIdentifiedRecognitions;
    // backup image and recognitions
    private Bitmap mPrevBitmap;
    private List<Classifier.Recognition> mPrevRecognitions;
    private List<Pair<Integer, String>> mPrevRecognitionTexts;

    private static final String TAG = "BookLabelDetector";
    private FirebaseVisionTextDetector mTextDetector;
    // Enlarge the detected bounding box by 20%
    private static final float BBOX_SCALE_FACTOR = 1.2f;

    // Variables for drawing on UI.
    // TODO: Decouple the drawing operations to another class.
    private List<Classifier.Recognition> mFrameRecognitions; // for display
    private int mFrameWidth;
    private int mFrameHeight;
    private int mSensorOrientation;
    private BorderedText mBorderedText;
    private BorderedText mBorderedPrompt;
    private final Paint mBoxPaint = new Paint();
    private Context mContext;

    public BookLabelDetector(Context context) {
        Log.v(TAG, "Constructor called.");

        mTextDetector = FirebaseVision.getInstance().getVisionTextDetector();
        mDetectionTask = TASK_NONE;
        mContext = context;

        mBoxPaint.setColor(Color.GREEN);
        mBoxPaint.setStyle(Paint.Style.STROKE);
        mBoxPaint.setStrokeWidth(16.0f); //12.0f
        mBoxPaint.setStrokeCap(Paint.Cap.ROUND);
        mBoxPaint.setStrokeJoin(Paint.Join.ROUND);
        mBoxPaint.setStrokeMiter(100);

        float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 14, context.getResources().getDisplayMetrics());
        mBorderedText = new BorderedText(textSizePx);

        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 26, context.getResources().getDisplayMetrics());
        mBorderedPrompt = new BorderedText(textSizePx);
        mBorderedPrompt.setTextAlign(Paint.Align.CENTER);
    }

    public void setDetectionTask(int task, String text) {
        String str;
        if (text == null)
            str = "(null)";
        else
            str = text;
        Log.v(TAG, "Received task: " + String.valueOf(task) + ", " + str);

        mDetectionTask = task;
        mDetectionText = Utils.stripString(text);
    }

    public void setDetectionText(String text) {
        mDetectionText = Utils.stripString(text);
        Log.v(TAG, "Detection text set to " + mDetectionText);
    }

    private void backupPreviousRecognitions() {
        if (mBitmap == null || mRecognitions == null || mRecognitions.isEmpty()) {
            Log.v(TAG, "No previous recognition to backup.");
            return;
        }

        mPrevBitmap = mBitmap;
        mPrevRecognitions = mRecognitions;
        mPrevRecognitionTexts = mRecognitionTexts;
    }

    public synchronized void processRecognitions(Bitmap rgbBitmap, List<Classifier.Recognition> recognitions,
                                                 List<Classifier.Recognition> frameRecognitions,
                                                 int frameWidth, int frameHeight, int sensorOrientation) {
        try {
            // Backup previous recognized labels for saving purpose.
            backupPreviousRecognitions();

            // Snapshot the current bitmap and recognitions.
            mBitmap = Bitmap.createBitmap(rgbBitmap);
            mRecognitions = recognitions;
            mFrameRecognitions = frameRecognitions;
            mFrameWidth = frameWidth;
            mFrameHeight = frameHeight;
            mSensorOrientation = sensorOrientation;

            Log.v(TAG, "Processing recognitions...");
            // TODO: filter bounding boxes.
            // TODO: sort bounding boxes.

            // Recognize label texts.
            recognizeTexts();

            if (mDetectionTask != TASK_NONE)
                performDetectionTask();

        } catch (Exception e) {
            Log.e(TAG, "Error processing recognition!", e);
        }
    }

    public synchronized void saveRecognitionImages() {
        if (mPrevBitmap == null || mPrevRecognitions == null || mPrevRecognitions.isEmpty()) {
            Log.d(TAG, "No image to save.");
            return;
        }

        try {
            Log.v(TAG, "Saving whole bitmap...");
            // Save the whole frame.
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String prefix = "booklabel_" + timeStamp + "_";
            String fileName = prefix + "whole_" + String.valueOf(mPrevBitmap.getWidth()) + "_" +
                    String.valueOf(mPrevBitmap.getHeight())+ ".jpg";
            // [HHJ] TODO: copy the bitmap and draw bounding boxes.
            Utils.saveBitmap(mPrevBitmap, fileName);

            if (mPrevRecognitions == null || mPrevRecognitions.isEmpty()) {
                Log.d(TAG, "No recognition to save.");
                return;
            }

            Log.v(TAG, "Saving " + String.valueOf(mPrevRecognitions.size()) + " recognitions to images...");

            // Save the individual book labels cropped out from the image.
            Bitmap croppedBitmap;
            int i = 0;

            for (Classifier.Recognition recognition: mPrevRecognitions) {
                RectF rectF = recognition.getLocation();
                // Enlarge the bounding box
                rectF = Utils.scaleRectF(rectF, BBOX_SCALE_FACTOR);
                croppedBitmap = Utils.cropBitmapFromRectF(mPrevBitmap, rectF);
                fileName = prefix + String.valueOf(i++) + ".jpg";
                Log.v(TAG, fileName);
                Utils.saveBitmap(croppedBitmap, fileName);
            }

            Log.v(TAG, "Finished saving recognitions to images.");
        } catch (Exception e) {
            Log.e(TAG, "Error saving recognition images!", e);
        }
    }

    public synchronized void saveRecognitionTexts() {
        if (mPrevRecognitionTexts == null || mPrevRecognitionTexts.isEmpty()) {
            return;
        }

        try {
            Log.v(TAG, "Saving " + String.valueOf(mPrevRecognitionTexts.size()) + " recognition texts.");

            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String fileName = "labeltext_" + timeStamp + ".txt";
            StringBuilder sb = new StringBuilder();
            for (Pair<Integer, String> pair: mPrevRecognitionTexts) {
                sb.append(pair.first.toString())
                        .append(": ")
                        .append(pair.second)
                        .append(System.getProperty("line.separator"));
            }
            Utils.saveText(sb.toString(), fileName);
            Log.v(TAG, "Finished saving recognition texts.");

            Toast.makeText(mContext,
                    "Finished saving "+ String.valueOf(mPrevRecognitionTexts.size()) + " recognition texts.",
                    Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "Error saving recognition texts.", e);
        }
    }

    /**
     * Perform text recognition based on the bitmap and detected bounding boxes of book labels.
     */
    private synchronized void recognizeTexts() {
        if (mBitmap == null || mRecognitions == null || mRecognitions.isEmpty()) {
            mRecognitionTexts = null;
            return;
        }

        Log.v(TAG, "Start recognizing " + String.valueOf(mRecognitions.size()) + " labels...");

        try {
            if (mRecognitionTexts == null)
                mRecognitionTexts = new ArrayList<>();
            else
                mRecognitionTexts.clear();

            Bitmap croppedBitmap;
            FirebaseVisionImage fvImage;

            Task<FirebaseVisionText> task;

            int i = 0;
            FirebaseVisionText result;
            long startTime, recognitionTime;
            for (Classifier.Recognition recognition : mRecognitions) {
                startTime = SystemClock.uptimeMillis();
                RectF rectF = recognition.getLocation();
                // Enlarge the bounding box
                rectF = Utils.scaleRectF(rectF, BBOX_SCALE_FACTOR);
                croppedBitmap = Utils.cropBitmapFromRectF(mBitmap, rectF);
                fvImage = FirebaseVisionImage.fromBitmap(croppedBitmap);

                task = mTextDetector.detectInImage(fvImage);

                result = Tasks.await(task, 1000, TimeUnit.MILLISECONDS);
                processTextRecognitionResult(i, result);
                i++;

                recognitionTime = SystemClock.uptimeMillis() - startTime;
                Log.v("[Profile][Rec]", String.valueOf(recognitionTime));
            }

            Log.v(TAG, "Finished recognizing label texts.");
        } catch (Exception e) {
            Log.e(TAG, "[Rec]Error recognizing texts!", e);
        }
    }

    /**
     * Draw recognized texts onto the canval.
     *
     * @param canvas Canvas to be drawn onto.
     */
    public synchronized void draw(final Canvas canvas) {
        try {
            if (mRecognitionTexts == null || mRecognitionTexts.isEmpty()) {
                return;
            }
            final boolean rotated = mSensorOrientation % 180 == 90;
            final float multiplier =
                    Math.min(canvas.getHeight() / (float) (rotated ? mFrameWidth : mFrameHeight),
                            canvas.getWidth() / (float) (rotated ? mFrameHeight : mFrameWidth));
            Matrix frameToCanvasMatrix =
                    ImageUtils.getTransformationMatrix(
                            mFrameWidth,
                            mFrameHeight,
                            (int) (multiplier * (rotated ? mFrameHeight : mFrameWidth)),
                            (int) (multiplier * (rotated ? mFrameWidth : mFrameHeight)),
                            mSensorOrientation,
                            false);


            // Draw task related prompts.
            drawPrompt(canvas, frameToCanvasMatrix);

            // Draw recognized texts.
            drawRecognizedText(canvas, frameToCanvasMatrix);


            Log.v(TAG, "Finished drawing texts.");
        } catch (Exception e) {
            Log.e(TAG, "Error when drawing on overlay!", e);
        }
    }

    private void processTextRecognitionResult(int index, FirebaseVisionText firebaseVisionText) {
        Log.v(TAG, "[Rec]Received text recognition for " + String.valueOf(index));

        // Merge texts from all blocks into a single string.
        StringBuilder sb = new StringBuilder();
        String text;
        for (FirebaseVisionText.Block block : firebaseVisionText.getBlocks()) {
            text = block.getText();
            sb.append(text).append("|");
        }
        text = sb.toString();

        mRecognitionTexts.add(new Pair<>(index, text));

        Log.v(TAG, "[Rec]Finished processing text: " + text);
    }

    /**
     * Perform user selected task.
     */
    private void performDetectionTask() {
        if (mRecognitionTexts == null || mRecognitionTexts.isEmpty())
            return;

        Log.v(TAG, "Start detection task." + mRecognitionTexts.size());
        mIdentifiedRecognitions = new ArrayList<>();
        switch (mDetectionTask) {
            case LOCATE_BOOK:
                if (mDetectionText == null || mDetectionText.isEmpty())
                    return;
                for (Pair<Integer, String> recgText: mRecognitionTexts) {
                    Log.v(TAG, "[Rec]Comparing " + mDetectionText + " and " + Utils.stripString(recgText.second));
                    if (mDetectionText.equalsIgnoreCase(Utils.stripString(recgText.second))) {
                        Log.v(TAG, "[Rec]Identified label " + mDetectionText);
                        mIdentifiedRecognitions.add(recgText.first);
                    }
                }
                break;
            case DETECT_MISPLACED:
                break;
            case INSERT_BOOK:
                break;
        }
        Log.v(TAG, "Finished detection task.");
    }

    private void drawRecognizedText(Canvas canvas, Matrix frameToCanvasMatrix) {
        Integer index;
        String text;
        Classifier.Recognition recognition;
        RectF location = new RectF();
        for (Pair<Integer, String> recgText : mRecognitionTexts) {
            // Get the position of the text.
            index = recgText.first;
            recognition = mFrameRecognitions.get(index);
            frameToCanvasMatrix.mapRect(location, recognition.getLocation());

            // Get the text.
            //text = "[" + index.toString() + "]" + Utils.stripString(recgText.second);
            text = Utils.stripString(recgText.second);
            mBorderedText.drawText(canvas, location.left, location.top-14, text);
        }
    }

    private void drawPrompt(Canvas canvas, Matrix frameToCanvasMatrix) {
        if (mDetectionTask != TASK_NONE &&
                mIdentifiedRecognitions != null &&
                !mIdentifiedRecognitions.isEmpty()) {

            Classifier.Recognition recognition;
            RectF location = new RectF();

            String[] prompts = mContext.getResources().getStringArray(R.array.task_prompts);
            String prompt = prompts[mDetectionTask];

            float x, y;
            float ex, ey;
            Rect rect = new Rect();
            for (Integer identified : mIdentifiedRecognitions) {
                recognition = mFrameRecognitions.get(identified);
                frameToCanvasMatrix.mapRect(location, recognition.getLocation());

                // Draw bounding box to highlight the detection.
                mBoxPaint.setColor(Color.GREEN);
                canvas.drawRect(location, mBoxPaint);

                // Draw the arrow
                x = location.centerX();
                ex = x;
                if (location.centerY() > canvas.getHeight()/2) {
                    y = Math.max(location.top - 50, 0f);
                    ey = Math.max(y - 50, 0f);
                } else {
                    y = Math.min(location.bottom + 50, canvas.getHeight());
                    ey = Math.min(y + 50, canvas.getHeight());
                }
                canvas.drawLine(x, y, ex, ey, mBoxPaint);

                // Draw the text for prompt.
                if (location.centerY() > canvas.getHeight()/2)
                    y = Math.max(location.top - 120, 20f);
                else {
                    mBorderedPrompt.getTextBounds(prompt, 0, prompt.length(), rect);
                    y = Math.min(location.bottom + rect.height() + 120, canvas.getHeight());
                }

                mBorderedPrompt.setInteriorColor(Color.GREEN);
                mBorderedPrompt.drawText(canvas, x, y, prompt);
            }
        }
    }
}
