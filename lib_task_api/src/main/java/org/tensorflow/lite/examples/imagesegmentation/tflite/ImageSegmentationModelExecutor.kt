/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Color.argb
import android.os.SystemClock
import android.util.Log
import kotlin.collections.HashMap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter
import org.tensorflow.lite.task.vision.segmenter.Segmentation

/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
 *
 * "s0", "s1", "s2", "s3", "s4", "s5", "s6",
 * "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
 * "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25"
 */
class ImageSegmentationModelExecutor(
  context: Context,
  private var useGPU: Boolean = true //false
) {

  private val imageSegmenter: ImageSegmenter

  private var fullTimeExecutionTime = 0L
  private var imageSegmentationTime = 0L
  private var maskFlatteningTime = 0L
  var Pokok = 0
  var Buah = 0
  var Sayur= 0
  var Lauk = 0

  init {
    //if (useGPU) {
    //  throw IllegalArgumentException("ImageSegmenter does not support GPU currently, but CPU.")
    //} else {
      imageSegmenter = ImageSegmenter.createFromFile(context, IMAGE_SEGMENTATION_MODEL)
    //}
  }

  fun execute(inputImage: Bitmap): ModelExecutionResult {
    try {
      fullTimeExecutionTime = SystemClock.uptimeMillis()

      imageSegmentationTime = SystemClock.uptimeMillis()
      val tensorImage = TensorImage.fromBitmap(inputImage)
      val results = imageSegmenter.segment(tensorImage)
      imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
      Log.d(TAG, "Time to run the ImageSegmenter $imageSegmentationTime")

      maskFlatteningTime = SystemClock.uptimeMillis()
      val (maskBitmap, itemsFound) = createMaskBitmapAndLabels(
        results.get(0), inputImage.getWidth(),
        inputImage.getHeight()
      )
      maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
      Log.d(TAG, "Time to create the mask and labels $maskFlatteningTime")

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")

      return ModelExecutionResult(
        /*bitmapResult=*/ stackTwoBitmaps(maskBitmap, inputImage),
        /*bitmapOriginal=*/ inputImage,
        /*bitmapMaskOnly=*/ maskBitmap,
        formatExecutionLog(inputImage.getWidth(), inputImage.getHeight()),
        itemsFound
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap =
        Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight(), Bitmap.Config.ARGB_8888)
      return ModelExecutionResult(
        emptyBitmap,
        emptyBitmap,
        emptyBitmap,
        exceptionLog,
        HashMap<String, Int>()
      )
    }
  }

  private fun createMaskBitmapAndLabels(
    result: Segmentation,
    inputWidth: Int,
    inputHeight: Int
  ): Pair<Bitmap, Map<String, Int>> {
    // For the sake of this demo, change the alpha channel from 255 (completely opaque) to 128
    // (semi-transparent), because the maskBitmap will be stacked over the original image later.
    val coloredLabels = result.getColoredLabels()
    var colors = IntArray(coloredLabels.size)
    var cnt = 0
    for (coloredLabel in coloredLabels) {
      val rgb = coloredLabel.getArgb()
      colors[cnt++] = Color.argb(ALPHA_VALUE, Color.red(rgb), Color.green(rgb), Color.blue(rgb))
    }
    Log.d(TAG,"numbers of label colors $cnt")
    // Use completely transparent for the background color.
    colors[0] = Color.TRANSPARENT

    // Create the mask bitmap with colors and the set of detected labels.
    val maskTensor = result.getMasks().get(0)
    val maskArray = maskTensor.getBuffer().array()
    val pixels = IntArray(maskArray.size)
    val itemsFound = HashMap<String, Int>()
    val warnaMerah = argb(128, 255,0,0).toInt()
    val warnaHijau = argb(128, 0,255,0).toInt()
    val warnaBiru = argb(128, 0,0,255).toInt()
    val warnaKuning = argb(128, 255,255,0).toInt()
    val warnaLain = argb(0, 0,0,0).toInt()
    var indeksMakanan = 0 //Pokok = 1, Lauk = 2, Sayur = 3, Buah = 4
    Pokok = 0
    Lauk = 0
    Sayur = 0
    Buah = 0
    for (i in maskArray.indices) {
      val cekLabel = coloredLabels.get(maskArray[i].toInt()).getlabel()
      var warna : Int = 0
      var namaJenis : String = ""
      if (cekLabel.contains("starch")) indeksMakanan = 1
      else if (cekLabel.contains("grains")) indeksMakanan = 1
      else if (cekLabel.contains("snack")) indeksMakanan = 1
      else if (cekLabel.contains("sweets")) indeksMakanan = 1
      else if (cekLabel.contains("other_food")) indeksMakanan = 1
      else if (cekLabel.contains("dairy")) indeksMakanan = 2
      else if (cekLabel.contains("herbs")) indeksMakanan = 2
      else if (cekLabel.contains("oils")) indeksMakanan = 2
      else if (cekLabel.contains("sauces")) indeksMakanan = 2
      else if (cekLabel.contains("soups")) indeksMakanan = 2
      else if (cekLabel.contains("protein")) indeksMakanan = 2
      else if (cekLabel.contains("vegetables")) indeksMakanan = 3
      else if (cekLabel.contains("fruits")) indeksMakanan = 4
      else if (cekLabel.contains("beverages")) indeksMakanan = 0
      else if (cekLabel.contains("dining")) indeksMakanan = 0
      else if (cekLabel.contains("container")) indeksMakanan = 0
      else indeksMakanan = 0

      if (indeksMakanan == 1 ){
        warna = warnaMerah
        namaJenis = "Pokok"
        Pokok++
      } else
      if (indeksMakanan == 2){
        warna = warnaKuning
        namaJenis = "Lauk"
        Lauk++
      } else
      if (indeksMakanan == 3){
        warna = warnaHijau
        namaJenis = "Sayur"
        Sayur++
      } else
      if (indeksMakanan == 4){
        warna = warnaBiru
        namaJenis = "Buah"
        Buah++
      } else {
        warna = warnaLain
        namaJenis = "Lain-lain"
      }

      pixels[i] = warna
      itemsFound.put(namaJenis, warna)
    }
    val maskBitmap = Bitmap.createBitmap(
      pixels, maskTensor.getWidth(), maskTensor.getHeight(),
      Bitmap.Config.ARGB_8888
    )
    // Scale the maskBitmap to the same size as the input image.
    return Pair(Bitmap.createScaledBitmap(maskBitmap, inputWidth, inputHeight, true), itemsFound)
  }
  /*
  private fun createMaskBitmapAndLabels(
    result: Segmentation,
    inputWidth: Int,
    inputHeight: Int
  ): Pair<Bitmap, Map<String, Int>> {
    // For the sake of this demo, change the alpha channel from 255 (completely opaque) to 128
    // (semi-transparent), because the maskBitmap will be stacked over the original image later.
    val coloredLabels = result.getColoredLabels()
    var colors = IntArray(coloredLabels.size)
    var cnt = 0
    for (coloredLabel in coloredLabels) {
      val rgb = coloredLabel.getArgb()
      colors[cnt++] = Color.argb(ALPHA_VALUE, Color.red(rgb), Color.green(rgb), Color.blue(rgb))
    }
    Log.d(TAG,"numbers of label colors $cnt")
    // Use completely transparent for the background color.
    colors[0] = Color.TRANSPARENT

    // Create the mask bitmap with colors and the set of detected labels.
    val maskTensor = result.getMasks().get(0)
    val maskArray = maskTensor.getBuffer().array()
    val pixels = IntArray(maskArray.size)
    val itemsFound = HashMap<String, Int>()
    for (i in maskArray.indices) {
      val color = colors[maskArray[i].toInt()]
      pixels[i] = color
      itemsFound.put(coloredLabels.get(maskArray[i].toInt()).getlabel(), color)
    }
    val maskBitmap = Bitmap.createBitmap(
      pixels, maskTensor.getWidth(), maskTensor.getHeight(),
      Bitmap.Config.ARGB_8888
    )
    // Scale the maskBitmap to the same size as the input image.
    return Pair(Bitmap.createScaledBitmap(maskBitmap, inputWidth, inputHeight, true), itemsFound)
  }
*/
  private fun stackTwoBitmaps(foregrand: Bitmap, background: Bitmap): Bitmap {
    val mergedBitmap =
      Bitmap.createBitmap(foregrand.getWidth(), foregrand.getHeight(), foregrand.getConfig())
    val canvas = Canvas(mergedBitmap)
    canvas.drawBitmap(background, 0.0f, 0.0f, null)
    canvas.drawBitmap(foregrand, 0.0f, 0.0f, null)
    return mergedBitmap
  }

  private fun formatExecutionLog(imageWidth: Int, imageHeight: Int): String {
    val sb = StringBuilder()
    val total:Int = (Pokok+Sayur+Lauk+Buah)
    val persenPokok = (Pokok*100)/total
    val persenSayur = (Sayur*100)/total
    val persenLauk = (Lauk*100)/total
    val persenBuah = (Buah*100)/total
    sb.append("Makanan Pokok: $persenPokok%\n")//"Input Image Size: $imageWidth x $imageHeight\n")
    sb.append("Lauk pauk    : $persenLauk%\n")//"ImageSegmenter execution time: $imageSegmentationTime ms\n")
    sb.append("Sayur-sayuran: $persenSayur%\n")//"GPU enabled: $useGPU\n")
    sb.append("Buah-buahan  : $persenBuah%\n")//Mask creation time: $maskFlatteningTime ms\n")
    if(persenPokok > 40) sb.append("Kurangi porsi Makanan Pokok!\n")//"Number of threads: $NUM_THREADS\n")
    if(persenLauk < 15)  sb.append("Tambahkan lagi porsi Lauk Pauk!\n")
    if(persenLauk > 40)  sb.append("Kurangi porsi Lauk Pauk!\n")
    if(persenSayur < 23) sb.append("Tambahkan lagi porsi Sayur-sayuran!\n")//"Number of threads: $NUM_THREADS\n")
    if(persenBuah < 15)  sb.append("Tambahkan lagi porsi Buah-buahan!\n")

    sb.append("\nWaktu analisa: $fullTimeExecutionTime ms\n")
    sb.append("                  Hasil deteksi>>>>>\n")
    return sb.toString()
  }

  fun close() {
    imageSegmenter.close()
  }

  companion object {
    public const val TAG = "SegmentationTask"
    private const val NUM_THREADS = 4
    private const val IMAGE_SEGMENTATION_MODEL = "lite-model_seefood_segmenter_mobile_food_segmenter_V1_1.tflite"
    private const val ALPHA_VALUE = 128
  }

  /*
  private fun estimateIsiPiringku(
    result: Segmentation
  ): Map<String, Int> {
    // For the sake of this demo, change the alpha channel from 255 (completely opaque) to 128
    // (semi-transparent), because the maskBitmap will be stacked over the original image later.
    val coloredLabels = result.getColoredLabels()
    var colors = IntArray(coloredLabels.size)
    var cnt = 0
    for (coloredLabel in coloredLabels) {

      val rgb = coloredLabel.getArgb()
      colors[cnt++] = Color.argb(ALPHA_VALUE, Color.red(rgb), Color.green(rgb), Color.blue(rgb))
    }

    // Use completely transparent for the background color.
    colors[0] = Color.TRANSPARENT

    // Create the mask bitmap with colors and the set of detected labels.
    // Pokok = starch/grains+snacks+sweets/desserts+other_food
    // Lauk = dairy+herbs/spices+fats/oils/sauces+soups/stews+protein
    // Sayur = vegetables
    // Buah = fruits
    val maskTensor = result.getMasks().get(0)
    val maskArray = maskTensor.getBuffer().array()
    val pixels = IntArray(maskArray.size)
    val itemsFound = HashMap<String, Int>()
    val warnaMerah = argb(128, 255,0,0).toInt()
    val warnaHijau = argb(128, 0,255,0).toInt()
    val warnaBiru = argb(128, 0,0,255).toInt()
    val warnaKuning = argb(128, 255,255,0).toInt()

    for (i in maskArray.indices) {
      val color = colors[maskArray[i].toInt()]
      val namajenis = coloredLabels.get(maskArray[i].toInt()).getlabel()
      if (namajenis.contains("starch")) namajenis = "Pokok"
      if (namajenis.contains("grains")) namajenis = "Pokok"
      if (namajenis.contains("snack")) namajenis = "Pokok"
      if (namajenis.contains("sweets")) namajenis = "Pokok"
      if (namajenis.contains("other")) namajenis = "Pokok"
      if (namajenis.contains("dairy")) namajenis = "Lauk"
      if (namajenis.contains("herbs")) namajenis = "Lauk"
      if (namajenis.contains("oils")) namajenis = "Lauk"
      if (namajenis.contains("sauces")) namajenis = "Lauk"
      if (namajenis.contains("soups")) namajenis = "Lauk"
      if (namajenis.contains("protein")) namajenis = "Lauk"
      if (namajenis.contains("vegetables")) namajenis = "Sayur"
      if (namajenis.contains("fruits")) namajenis = "Buah"

      if (namajenis == "Pokok") color = warnaMerah
      if (namajenis == "Lauk") color = warnaKuning
      if (namajenis == "Sayur") color = warnaHijau
      if (namajenis == "Buah") color = warnaBiru

      pixels[i] = color
      itemsFound.put(namajenis, color)
    }

    return itemsFound
  }*/
}
