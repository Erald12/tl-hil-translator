plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.example.tlhiltranslator"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.tlhiltranslator"
        minSdk = 24                       // Chaquopy's sentencepiece wheel needs API 24+
        targetSdk = 35
        // Each GitHub build gets a higher number, so a new APK installs over the old one
        versionCode = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 1
        versionName = "1.0." + (System.getenv("GITHUB_RUN_NUMBER") ?: "0")
        ndk { abiFilters += listOf("arm64-v8a") }   // vivo Y27 = 64-bit ARM
    }

    // full = phrase list + AI models (~600 MB)   lite = phrase list only (~30 MB)
    flavorDimensions += "size"
    productFlavors {
        create("full") {
            dimension = "size"
            manifestPlaceholders["appLabel"] = "Tagalog-Hiligaynon"
        }
        create("lite") {
            dimension = "size"
            applicationIdSuffix = ".lite"        // both versions can be installed side by side
            manifestPlaceholders["appLabel"] = "TL-HIL Lite"
        }
    }

    // Models and databases are stored uncompressed (faster first-launch copy)
    androidResources { noCompress += listOf("onnx", "spm", "db") }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

chaquopy {
    defaultConfig {
        version = "3.10"
    }
    productFlavors {
        getByName("full") {
            pip { install("sentencepiece") }    // tokenizer for the AI models
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    "fullImplementation"("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
}
