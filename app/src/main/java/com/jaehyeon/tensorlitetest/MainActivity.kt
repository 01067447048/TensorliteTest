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

        imagePickerLauncher = registerImagePicker { result ->
            binding.ivSelectPic.setImageURI(result.lastOrNull()?.uri)
            viewModel.addImage(result)
        }

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
        lifecycleScope.launchWhenResumed {
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

    private fun initModel() {
        model = ClassifyModel(assets, ClassifyModel.TEST_MODEL, applicationContext)
        model.init()
    }

    override fun onDestroy() {
        if (::model.isInitialized) model.finish()
        super.onDestroy()
    }
}