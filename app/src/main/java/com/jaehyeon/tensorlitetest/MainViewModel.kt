package com.jaehyeon.tensorlitetest

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.lifecycle.ViewModel
import com.esafirm.imagepicker.model.Image
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Created by Jaehyeon on 2022/10/13.
 */
class MainViewModel: ViewModel() {
    /**
     * viewmodel 에서 관리할 Image
     */
    private val _images = MutableStateFlow<Image?>(null)
    val images: StateFlow<Image?> = _images.asStateFlow()

    /**
     * add Image
     * @param result Image list
     */
    fun addImage(result: List<Image>) {
        _images.value = result.lastOrNull()
    }
}

/**
 * Image 파일을 Bitmap 으로 변환.
 */
fun Image.toBitmap(): Bitmap {
    return BitmapFactory.decodeFile(this.path)
}