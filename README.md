# Image Segmentation Android sample with added IsiPiringku estimator

The used model, SeeFoodDeepLab
[https://tfhub.dev/google/lite-model/seefood/segmenter/mobile_food_segmenter_V1/1] is a
deep learning model for semantic food image segmentation, where the goal
is to assign semantic food labels (e.g. fruits, beverages) to every pixel in the input
image.

## Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)
*   An Android device, or an Android Emulator

## Build and run

### Step 1. Clone this examples source code

### Step 2. Import the sample app to Android Studio

### Step 3. Run the Android app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

#### Switch between inference solutions (Task library vs TFLite Interpreter)

This image segmentation Android reference app demonstrates two implementation
solutions:

(1)
[`lib_task_api`](./image_segmentation/android/lib_task_api)
that leverages the out-of-box API from the
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter);

(2)
[`lib_interpreter`](./image_segmentation/android/lib_interpreter)
that creates the custom inference pipleline using the
[TensorFlow Lite Interpreter Java API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_java).

The [`build.gradle`](app/build.gradle) inside `app` folder shows how to change
`flavorDimensions "tfliteInference"` to switch between the two solutions.

Inside **Android Studio**, you can change the build variant to whichever one you
want to build and run — just go to `Build > Select Build Variant` and select one
from the drop-down menu. See
[configure product flavors in Android Studio](https://developer.android.com/studio/build/build-variants#product-flavors)
for more details.

To test the app, open the app called `Isi Piringku Estimator Proporsi` on your device.
Re-installing the app may require you to uninstall the previous installations.

For gradle CLI, running `./gradlew build` can create APKs for both solutions
under `app/build/outputs/apk`.

*Note: If you simply want the out-of-box API to run the app, we recommend
`lib_task_api` for inference. If you want to customize your own models and
control the detail of inputs and outputs, it might be easier to adapt your model
inputs and outputs by using `lib_interpreter`.*

## Resources used:

*   Camera2:
    https://developer.android.com/reference/android/hardware/camera2/package-summary
*   Camera2 base sample:
    https://github.com/android/camera-samples/tree/master/Camera2Formats
*   TensorFlow Lite: https://www.tensorflow.org/lite
*   ImageSegmentation model:
    https://www.tensorflow.org/lite/models/segmentation/overview
