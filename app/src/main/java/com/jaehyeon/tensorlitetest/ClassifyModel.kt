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

/******
 * 분류 모델.
 * 모델이 저장 돼 있는 assets.
 * model 이름
 * context
 */
class ClassifyModel(
    private val assetManager: AssetManager,
    private val modelName: String,
    private val context: Context
) {

    // 모델을 실행 시킬 interpreter
    private lateinit var interpreter: Interpreter
    // 모델의 색 채널을 확인.
    private var modelInputChannel = 0
    // 모델에서 받을 사진의 너비
    private var modelInputWidth = 0
    // 모델에서 받을 사진의 높이
    private var modelInputHeight = 0
    // 모델이 분류할 클래스의 개수
    private var modelOutputClasses = 0
    //Tensor 에서 사용할 이미지.
    private lateinit var inputImage : TensorImage
    // 실제 모델에 넣을 이미지의 byte buffer
    private lateinit var outputBuffer: TensorBuffer
    // 레이블을 정리 갖고 있을 리스트.
    private val labels = mutableListOf<String>()

    /**
     * 모델 초기화 함수
     */
    fun init() {
        val model = loadModelFile()
        model?.let {
            it.order(ByteOrder.nativeOrder()) // model 을 cpu 가 잘 읽을 수 있는 순서로 정렬.
            interpreter = Interpreter(model) // 인터프리터가 현재 모델을 해석 할 수 있도록 함.
            initModelShape() // 모델 쉐이프 정리.

            val outputTensor = interpreter.getOutputTensor(0) // 모델 결과에
            val outputShape = outputTensor.shape() // output 의 모양 정리, [1]
            modelOutputClasses = outputShape[1] // 모델의 아웃풋 클래스 정의.
            labels.addAll(FileUtil.loadLabels(context, LABEL_FILE)) // 레이블 업로드
        }
    }

    /**
     * 모델 파일 불러옴.
     */
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

    /**
     * 모델 인풋, 아웃풋 정의.
     */
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

    /**
     * image 를 tensor lite 가 해석 할 수 있는
     * tensor image 로 변환.
     */
    private fun loadImage(bitmap: Bitmap): TensorImage {
        inputImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(modelInputWidth, modelInputHeight, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        return imageProcessor.process(inputImage)
    }

    /**
     * 실제로 분류.
     */
    fun classify(image: Bitmap): Pair<String, Float> {
        inputImage = loadImage(image) // image
        interpreter.run(inputImage.buffer, outputBuffer.buffer.rewind()) // 실제 이미지, 그 결과를 담을 버퍼.
        // 매핑
        val output = TensorLabel(labels, outputBuffer).mapWithFloatValue // 실제 결과가 담긴 변수
        /**
         * 추가.
         * 텐서는 모든 것을 확률로 본다.
         * 예.
         * 0 class : 0.1 > 10%
         * 1 class : 0.9 > 90%
         */
        return argmax(output)
    }

    /**
     * 확률로 정의 된 것 중 확률이 가장 큰 class 가 실제 모델이 판별 한 것으로 보고
     * 그것을 return 함.
     */
    private fun argmax(map: Map<String, Float>) =
        map.entries.maxByOrNull { it.value }?.let {
            it.key to it.value
        } ?: ("" to 0f)

    /**
     * 종료시 자원 해제.
     */
    fun finish() {
        if (::interpreter.isInitialized) interpreter.close()
    }

    companion object {
        const val TEST_MODEL = "mask_classification_model.tflite"
        const val LABEL_FILE = "labels.txt"
    }
}