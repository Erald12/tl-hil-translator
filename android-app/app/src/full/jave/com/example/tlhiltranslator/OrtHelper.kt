package com.example.tlhiltranslator

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer

/**
 * Thin wrapper over ONNX Runtime Android, called from translator.py (ChaquopyBackend).
 * Keeps the encoder output and the large logits arrays on the Java side; Python only
 * passes token-id arrays in and gets one token id back per step.
 */
class OrtHelper(encoderPath: String, decoderPath: String, threads: Int) {
    private val env = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val encoder = env.createSession(encoderPath, opts)
    private val decoder = env.createSession(decoderPath, opts)

    private var encResult: OrtSession.Result? = null
    private var encHidden: OnnxTensor? = null
    private var encMask: OnnxTensor? = null

    @Synchronized
    fun encode(ids: LongArray) {
        releaseEncoderState()
        val shape = longArrayOf(1, ids.size.toLong())
        val mask = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { 1L }), shape)
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { idT ->
            val r = encoder.run(mapOf("input_ids" to idT, "attention_mask" to mask))
            encResult = r
            encHidden = r.get(0) as OnnxTensor
        }
        encMask = mask
    }

    /** Greedy step: returns argmax token id of the last position, skipping bannedId (pad). */
    @Synchronized
    fun nextToken(prefix: LongArray, bannedId: Int): Int {
        val shape = longArrayOf(1, prefix.size.toLong())
        OnnxTensor.createTensor(env, LongBuffer.wrap(prefix), shape).use { idT ->
            decoder.run(
                mapOf(
                    "input_ids" to idT,
                    "encoder_attention_mask" to encMask!!,
                    "encoder_hidden_states" to encHidden!!,
                )
            ).use { r ->
                val logitsT = r.get(0) as OnnxTensor           // [1, T, vocab]
                val vocab = logitsT.info.shape[2].toInt()
                val buf = logitsT.floatBuffer
                val offset = (prefix.size - 1) * vocab
                var best = 0
                var bestVal = Float.NEGATIVE_INFINITY
                for (i in 0 until vocab) {
                    if (i == bannedId) continue
                    val v = buf.get(offset + i)
                    if (v > bestVal) { bestVal = v; best = i }
                }
                return best
            }
        }
    }

    private fun releaseEncoderState() {
        encResult?.close(); encMask?.close()
        encResult = null; encHidden = null; encMask = null
    }

    fun close() {
        releaseEncoderState()
        encoder.close(); decoder.close()
    }
}
