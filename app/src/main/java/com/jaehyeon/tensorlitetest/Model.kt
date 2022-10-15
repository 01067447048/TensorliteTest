package com.jaehyeon.tensorlitetest

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel


/**
 * Created by Jaehyeon on 2022/10/13.
 */
class Model(
    private val assetManager: AssetManager,
    private val modelName: String,
    private val context: Context
) {

    lateinit var interpreter: Interpreter
    private var modelInputChannel = 0
    private var modelInputWidth = 0
    private var modelInputHeight = 0
    private var modelOutputClasses = 0
    private lateinit var inputImage : TensorImage
    private lateinit var outputBuffer: TensorBuffer
    private val labels = mutableListOf<String>()

    fun init() {
        val model = loadModelFile()
        model?.let {
            it.order(ByteOrder.nativeOrder())
            interpreter = Interpreter(model)
            initModelShape()

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            modelOutputClasses = outputShape[1]
            labels.addAll(FileUtil.loadLabels(context, LABEL_FILE))
        }
    }

    private fun loadImage(bitmap: Bitmap): TensorImage {
        inputImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputWidth, modelInputHeight, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        return imageProcessor.process(inputImage)
    }

    private fun loadModelFile(): ByteBuffer? {
        return try {
            val assetFileDescriptor = assetManager.openFd(modelName)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (t: Throwable) {
            Log.e(javaClass.simpleName, "loadModelFile: ${t.localizedMessage}")
            null
        }
    }

    private fun initModelShape() {
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        modelInputChannel = inputShape[3]
        modelInputWidth = inputShape[1]
        modelInputHeight = inputShape[2]

        inputImage = TensorImage(inputTensor.dataType())
        val outputTensor = interpreter.getOutputTensor(0)
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())
        //[1, 224, 224, 3]
    }

    private fun resizeBitmap(bitmap: Bitmap) =
        Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight, false)

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(bitmap.byteCount * modelInputChannel)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(bitmap.width * bitmap.height * modelInputChannel)

        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        pixels.forEach { pixel ->
            val r = pixel shr 16 and 0XFF
            val g = pixel shr 8 and 0xFF
            val b = pixel and 0xFF

            byteBuffer.put(((r - IMAGE_MEAN) / IMAGE_STD * 255).toInt().toByte())
            byteBuffer.put(((g - IMAGE_MEAN) / IMAGE_STD * 255).toInt().toByte())
            byteBuffer.put(((b - IMAGE_MEAN) / IMAGE_STD * 255).toInt().toByte())
        }
        return byteBuffer
    }

    private fun convertBitmapGrayByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(bitmap.byteCount)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        pixels.forEach { pixel ->
            val r = pixel shr 16 and 0xFF
            val g = pixel shr 8 and 0xFF
            val b = pixel and 0xFF

            val avgPixelValue = (r + g + b) / 3.0f
            val normalizedPixelValue = avgPixelValue / 255.0f

            byteBuffer.putFloat(normalizedPixelValue)
        }
        return byteBuffer
    }

    fun classify(image: Bitmap): Pair<Int, Float> {
//        val buffer = convertBitmapGrayByteBuffer(resizeBitmap(image))
        val buffer = convertBitmapToByteBuffer(resizeBitmap(image))
        val result = Array(1) { FloatArray(modelOutputClasses) { 0f } }
//        val result = Array(1) { Array(modelOutputClasses) { "" } }
        interpreter.run(buffer, result)
        return argmax(result[0])
    }

    private fun argmax(array: Array<String>): Pair<Int, Float> {
        var maxIndex = 0
        var maxValue = 0f
        Log.e("TAG", "argmax: ${array[0]}", )
//        array.forEachIndexed { index, value ->
//            if (value > maxValue) {
//                maxIndex = index
//                maxValue = value
//            }
//        }
        return maxIndex to maxValue
    }

    private fun argmax(array: FloatArray): Pair<Int, Float> {
        var maxIndex = 0
        var maxValue = 0f
        array.forEachIndexed { index, value ->
            if (value > maxValue) {
                maxIndex = index
                maxValue = value
            }
        }
        return maxIndex to maxValue
    }

    fun finish() {
        if (::interpreter.isInitialized) interpreter.close()
    }

    companion object {
        const val TEST_MODEL = "mask_classification_model.tflite"
        const val LABEL_FILE = "labels.txt"
        private const val IMAGE_MEAN = 128
        private const val IMAGE_STD = 128.0f
    }
}