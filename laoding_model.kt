private fun loadModel(): Interpreter {
        val fd = context.assets.openFd("mobilenet_ssd_quantized.tflite")
        val inputStream = FileInputStream(fd.fileDescriptor)
        val fileChannel = inputStream.channel
        return Interpreter(
            fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        )
    }
