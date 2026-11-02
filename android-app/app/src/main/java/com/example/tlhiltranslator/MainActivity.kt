package com.example.tlhiltranslator

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.pm.PackageInfoCompat
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Translation order:
 *   1. translations.db phrase list (fuzzy match, instant, hand-written entries)
 *   2. AI models (translator.py + ONNX Runtime) if no close phrase match
 * Layout: res/layout/activity_main.xml (input, btnTglHil, btnHilTgl, output, source).
 */
class MainActivity : AppCompatActivity() {
    private var translator: PyObject? = null
    private var phrases: PyObject? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val input = findViewById<EditText>(R.id.input)
        val output = findViewById<TextView>(R.id.output)
        val source = findViewById<TextView>(R.id.source)
        val btnTglHil = findViewById<Button>(R.id.btnTglHil)
        val btnHilTgl = findViewById<Button>(R.id.btnHilTgl)
        btnTglHil.isEnabled = false
        btnHilTgl.isEnabled = false

        output.text = "First launch: preparing offline files (about 1 minute)..."
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) { copyAssetsOnce() }
                withContext(Dispatchers.Default) {
                    if (!Python.isStarted()) Python.start(AndroidPlatform(this@MainActivity))
                    val py = Python.getInstance()
                    phrases = py.getModule("database_builder")
                        .callAttr("PhraseSearcher", File(filesDir, "translations.db").absolutePath)
                    val modelsDir = File(filesDir, "models")
                    if (File(modelsDir, "tl-en/meta.json").exists()) {
                        translator = py.getModule("translator")
                        translator!!.callAttr("init", modelsDir.absolutePath, 4)
                    }
                }
                output.text = if (translator != null) "Ready." else "Ready (phrase list only, no AI models found)."
                btnTglHil.isEnabled = true
                btnHilTgl.isEnabled = true
            } catch (e: Exception) {
                output.text = "Startup error: ${e.message}"
            }
        }

        fun run(src: String, tgt: String) {
            val text = input.text.toString().trim()
            if (text.isEmpty()) return
            output.text = "Translating..."
            source.text = ""
            lifecycleScope.launch {
                val (result, from) = withContext(Dispatchers.Default) {
                    try {
                        val dir = if (src == "tgl") "tgl2hil" else "hil2tgl"
                        val hit = phrases?.callAttr("best", text, dir, 85.0)
                        when {
                            hit != null -> hit.toString() to "phrase list"
                            translator != null ->
                                translator!!.callAttr("translate", text, src, tgt).toString() to "AI model (via English)"
                            else -> "No match in phrase list." to ""
                        }
                    } catch (e: Exception) {
                        "Error: ${e.message}" to ""
                    }
                }
                output.text = result
                source.text = if (from.isNotEmpty()) "Source: $from" else ""
            }
        }
        btnTglHil.setOnClickListener { run("tgl", "hil") }
        btnHilTgl.setOnClickListener { run("hil", "tgl") }
    }

    /** Native code needs real file paths, so copy assets -> internal storage once per app version. */
    private fun copyAssetsOnce() {
        val version = PackageInfoCompat.getLongVersionCode(packageManager.getPackageInfo(packageName, 0))
        val marker = File(filesDir, ".assets_v$version")
        if (marker.exists()) return
        copyAssetTree("", filesDir)
        marker.createNewFile()
    }

    private fun copyAssetTree(path: String, dest: File) {
        val children = assets.list(path).orEmpty()
        if (children.isEmpty()) {                       // it's a file
            if (path.isEmpty()) return
            assets.open(path).use { inp -> dest.outputStream().use { inp.copyTo(it, 1 shl 20) } }
            return
        }
        for (child in children) {
            if (path.isEmpty() && child !in setOf("models", "translations.db", "corpus.db")) continue
            val childPath = if (path.isEmpty()) child else "$path/$child"
            val childDest = File(dest, child)
            if (assets.list(childPath).orEmpty().isNotEmpty()) childDest.mkdirs()
            copyAssetTree(childPath, childDest)
        }
    }
}
