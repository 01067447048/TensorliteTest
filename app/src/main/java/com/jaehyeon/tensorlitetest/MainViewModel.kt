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

//    private val images = arrayListOf<Image>()

    private val _images = MutableStateFlow<Image?>(null)
    val images: StateFlow<Image?> = _images.asStateFlow()

    fun addImage(result: List<Image>) {
        _images.value = result.lastOrNull()
    }
}

fun Image.toBitmap(): Bitmap {
    return BitmapFactory.decodeFile(this.path)
}