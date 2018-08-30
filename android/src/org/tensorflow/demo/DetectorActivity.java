/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.PersistableBundle;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import org.librarybook.app.BookLabelDetector;
import org.librarybook.app.EnterLabelFragment;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
//import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.
import org.librarybook.app.R; // Explicit import needed for internal Google builds.

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener,
        AdapterView.OnItemSelectedListener,
        EnterLabelFragment.OnFragmentInteractionListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO;
  }
  private static final DetectorMode MODE = DetectorMode.YOLO; //TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  // [HHJ] TODO: Adjust confidence threshold.
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.6f; // 0.25f;

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  // [HHJ]
    private long lastRecognitionTimeMs;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  // [HHJ] Add book label detector
  private static BookLabelDetector bookLabelDetector;
  private static Spinner spinner;
  private EnterLabelFragment dialog;
  private static String labelText;
  private static final String KEY_TASK = "Detection Task";
  private static final String KEY_LABELTEXT = "Book Label Text";

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    // [HHJ] Add callback for book label detector.

      trackingOverlay.addCallback(new DrawCallback() {
          @Override
          public void drawCallback(Canvas canvas) {
              bookLabelDetector.draw(canvas);
          }
      });

    // Add callback for debug overlay.
    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");
            // [HHJ] Include text recognition time
              lines.add("Recognition time: " + lastRecognitionTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });

    // [HHJ] save detections when long clicked.

    trackingOverlay.setOnLongClickListener(new View.OnLongClickListener() {
        @Override
        public boolean onLongClick(View view) {
            LOGGER.v("Tracking overlay clicked and held.");
            runInBackground(new Runnable() {
                @Override
                public void run() {
                    bookLabelDetector.saveRecognitionImages();
                    bookLabelDetector.saveRecognitionTexts();
                }
            });
            return true;
        }
    });

    initializeSpinner();

  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            // [HHJ] Log for profiling.
              LOGGER.i("[Profile][Det]: " + lastProcessingTimeMs);

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
            }


            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            // [HHJ]
              final List<Classifier.Recognition> filteredRecognitions = new ArrayList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                // canvas.drawRect(location, paint);

                // [HHJ] Modify below to copy a recognition instead of modifying the original result.
                //cropToFrameTransform.mapRect(location);
                //result.setLocation(location);
                //mappedRecognitions.add(result);

                  filteredRecognitions.add(result);

                  RectF mappedLocation = new RectF();
                  cropToFrameTransform.mapRect(mappedLocation, location);
                  Classifier.Recognition mappedResult = new Classifier.Recognition(
                          result.getId(), result.getTitle(), result.getConfidence(), mappedLocation);
                  mappedRecognitions.add(mappedResult);

              }
            }

              // [HHJ] process text recognition before redrawing the overlay.
              // bookLabelDetector.processRecognitions(rgbFrameBitmap, mappedRecognitions);
              startTime = SystemClock.uptimeMillis();
              bookLabelDetector.processRecognitions(cropCopyBitmap, filteredRecognitions,
                      mappedRecognitions, previewWidth, previewHeight, sensorOrientation);
              lastRecognitionTimeMs = SystemClock.uptimeMillis() - startTime;

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();



            requestRender();
            computingDetection = false;
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }

  // [HHJ] Add spinner handling.

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
      if (i == BookLabelDetector.LOCATE_BOOK || i == BookLabelDetector.INSERT_BOOK){
        if (dialog == null)
            dialog = new EnterLabelFragment();
        dialog.show(getSupportFragmentManager(), getString(R.string.enter_label));
      }

          bookLabelDetector.setDetectionTask(i, labelText);
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {}

    // [HHJ] Add dialog handling.

    @Override
    public void onFragmentInteraction(String text) {
        labelText = text;
        bookLabelDetector.setDetectionText(text);
    }

    // [HHJ] Persist states.

    @Override
    public void onSaveInstanceState(Bundle outState) {
        int task = 0;
        if (spinner != null)
            task = spinner.getSelectedItemPosition();
        LOGGER.d("Saving instance state. " + task);
        outState.putInt(KEY_TASK, task);
        outState.putString(KEY_LABELTEXT, labelText);

        super.onSaveInstanceState(outState);

    }

    @Override
    public void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);

        int task = savedInstanceState.getInt(KEY_TASK);
        if (spinner != null) {
            LOGGER.d("Restoring spinner state. " + task);
            spinner.setSelection(task);
        } else
            LOGGER.d("Failed to restore spinner." + task);

        labelText = savedInstanceState.getString(KEY_LABELTEXT);
        LOGGER.d("Restoring booklabelDetector.");
        bookLabelDetector.setDetectionTask(task, labelText);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LOGGER.d("onCreate");

        if (bookLabelDetector == null)
            bookLabelDetector = new BookLabelDetector(this);


    }

    private void initializeSpinner() {
        spinner = (Spinner) findViewById(R.id.sp_task);
        if (spinner != null) {
            ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.detection_tasks, R.layout.spinner_select_task);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinner.setAdapter(adapter);
            spinner.setOnItemSelectedListener(this);
        }
    }

}
