package com.jaehyeon.tensorlitetest

import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import androidx.lifecycle.lifecycleScope
import com.esafirm.imagepicker.features.ImagePickerConfig
import com.esafirm.imagepicker.features.ImagePickerLauncher
import com.esafirm.imagepicker.features.ImagePickerMode
import com.esafirm.imagepicker.features.registerImagePicker
import com.jaehyeon.tensorlitetest.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.collectLatest

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var model: ClassifyModel
    private var imagePickerLauncher: ImagePickerLauncher? = null
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main)
        initModel()

        /**
         * image 선택 하면 선택 한 image 를 viewmodel로 전달하고
         * ImageView 에 선택한 이미지를 보여 줌.
         */
        imagePickerLauncher = registerImagePicker { result ->
            binding.ivSelectPic.setImageURI(result.lastOrNull()?.uri)
            viewModel.addImage(result)
        }

        /**
         * button 클릭 시 imagePickerLauncher 실행하여 이미지를 선택.
         * 사진은 1개만 가져옴.
         */
        binding.btnPichture.setOnClickListener {
            val config = ImagePickerConfig(
                mode = ImagePickerMode.SINGLE,
                theme = R.style.Theme_TensorliteTest
            )
            imagePickerLauncher?.launch(config)
        }
    }

    override fun onResume() {
        super.onResume()
        /**
         * viewmodel 에서 이미지를 flow 로 관리 하기 때문에 flow 처리를 위한 Coroutine Scope
         */
        lifecycleScope.launchWhenResumed {
            /**
             * flow 에서 사진이 업로드 될 시 모델에서 분류.
             */
            viewModel.images.collectLatest { image ->
                image?.let {
                    try {
                        val result = model.classify(image.toBitmap())
                        binding.tvResult.text = "${result.first} / ${result.second * 100.0f}%"
                    } catch (t: Throwable) {
                        Log.e(javaClass.simpleName, "classify: ${t.cause?.localizedMessage}", )
                    }

                }
            }
        }
    }

    /**
     * Activity 에서 ClassifyModel 을 사용 하기 위해 초기화 하는 함수.
     */

    private fun initModel() {
        model = ClassifyModel(assets, ClassifyModel.TEST_MODEL, applicationContext)
        model.init()
    }

    /**
     * 종료시 Interpreter 자원 해제.
     */
    override fun onDestroy() {
        model.finish()
        super.onDestroy()
    }
}