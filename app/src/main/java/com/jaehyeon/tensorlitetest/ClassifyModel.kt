package com.jaehyeon.tensorlitetest

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Created by Jaehyeon on 2022/10/15.
 */
class ClassifyModel(
    private val assetManager: AssetManager,
    private val modelName: String,
    private val context: Context
) {

    private lateinit var interpreter: Interpreter
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

    private fun loadImage(bitmap: Bitmap): TensorImage {
        inputImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputWidth, modelInputHeight, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        return imageProcessor.process(inputImage)
    }

    fun classify(image: Bitmap): Pair<String, Float> {
        inputImage = loadImage(image)
        interpreter.run(inputImage.buffer, outputBuffer.buffer.rewind())
        // 매핑
        val output = TensorLabel(labels, outputBuffer).mapWithFloatValue
        return argmax(output)
    }

    private fun argmax(map: Map<String, Float>) =
        map.entries.maxByOrNull { it.value }?.let {
            it.key to it.value
        } ?: ("" to 0f)

    fun finish() {
        if (::interpreter.isInitialized) interpreter.close()
    }

    companion object {
        const val TEST_MODEL = "mask_classification_model.tflite"
        const val LABEL_FILE = "labels.txt"
    }
}