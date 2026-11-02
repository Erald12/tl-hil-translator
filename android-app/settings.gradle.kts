pluginManagement {
    repositories {
        google()
        mavenCentral()          // Chaquopy plugin lives here
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "TLHilTranslator"
include(":app")
