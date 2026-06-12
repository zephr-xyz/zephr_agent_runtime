import java.util.Properties
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.plugin.serialization")
    id("maven-publish")
}

val platformVersions: CharSequence = providers.fileContents(rootProject.layout.projectDirectory.file("../platform_versions.json"))
    .asText
    .get()

abstract class SyncAssetDirectory : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceDir: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun sync() {
        val output = outputDir.get().asFile
        project.delete(output)
        project.copy {
            from(sourceDir)
            into(output)
        }
    }
}

abstract class CopyLiteRtSharedLibs : DefaultTask() {
    @get:Input
    abstract val nativeDepsRoot: Property<String>

    @get:Input
    abstract val nativeConfig: Property<String>

    @get:Input
    abstract val abis: ListProperty<String>

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copyLibraries() {
        val output = outputDir.get().asFile
        project.delete(output)
        for (abi in abis.get()) {
            val libDir = project.file("${nativeDepsRoot.get()}/android/2_installed/${nativeConfig.get()}/$abi/litert/lib")
            project.copy {
                from(libDir) {
                    include("libLiteRt.so")
                    include("libLiteRtClGlAccelerator.so")
                    include("libLiteRtDispatch_GoogleTensor.so")
                    into(abi)
                }
                into(output)
            }
        }
    }
}

fun platformVersion(key: String): String {
    val pattern = Regex("\"$key\"\\s*:\\s*\"([^\"]+)\"")
    return pattern.find(platformVersions)?.groupValues?.get(1)
        ?: error("platform_versions.json missing android.$key")
}

val nativeAbis = listOf("arm64-v8a")
val repoRoot = rootProject.layout.projectDirectory.asFile.parentFile
val projectPython = repoRoot.resolve(".venv/bin/python")
if (!projectPython.canExecute()) {
    error("Project Python not found at ${projectPython.absolutePath}. Run: uv sync")
}

providers.exec {
    workingDir = repoRoot
    commandLine(projectPython.absolutePath, "-m", "clis.support.dev_environment.native_deps_guard", "--bridge", "properties")
}.standardOutput.asText.get()

val nativeDepsProperties = Properties().apply {
    val file = rootProject.layout.projectDirectory.file("../1_build/native_deps/native_deps.properties").asFile
    if (!file.isFile) {
        error("Native deps bridge missing at ${file.absolutePath}. Run: uv run prepare_dev")
    }
    file.inputStream().use { load(it) }
}
val nativeDepsRootPath = nativeDepsProperties.getProperty("ZEPHR_NATIVE_DEPS_ROOT")
    ?: error("ZEPHR_NATIVE_DEPS_ROOT missing from native deps bridge. Run: uv run prepare_dev")
val nativeDepsRecipeHash = nativeDepsProperties.getProperty("ZEPHR_NATIVE_DEPS_RECIPE_HASH")
    ?: error("ZEPHR_NATIVE_DEPS_RECIPE_HASH missing from native deps bridge. Run: uv run prepare_dev")

fun parseVersionFromGithubTag(tag: String): String {
    val trimmed = tag.trim()
    if (trimmed.isEmpty()) {
        return "0.0.3-SNAPSHOT"
    }
    val match = Regex(""".*?(\d+\.\d+\.\d+)$""").matchEntire(trimmed)
        ?: error("github.tag '$trimmed' must end with a semantic version like 0.1.2")
    return match.groupValues[1]
}

val sdkPrivateGroupIdSuffix = providers.gradleProperty("zephrGroupIdSuffix")
    .orElse(providers.gradleProperty("zephr_developer_name"))
    .map { it.trim().ifEmpty { "unknown_buildenv" } }
    .orElse("unknown_buildenv")
val sdkGroupId = providers.gradleProperty("zephrSdkGroupId")
    .orElse(sdkPrivateGroupIdSuffix.map { "xyz.zephr.sdks.agent.$it" })
val sdkArtifactId = providers.gradleProperty("zephrSdkArtifactId")
    .orElse("zephr-agent-runtime")
val sdkVersion = providers.gradleProperty("zephrSdkVersion")
    .orElse(providers.gradleProperty("github.tag").map(::parseVersionFromGithubTag))
    .orElse("0.0.3-SNAPSHOT")
val sdkPrivateMavenUrl = providers.gradleProperty("zephrPrivateMavenUrl")
    .orElse("artifactregistry://us-central1-maven.pkg.dev/zephr-xyz-firebase-development/maven-repo")

android {
    namespace = "xyz.zephr.sdks.agent"
    compileSdk = platformVersion("compile_sdk").toInt()
    buildToolsVersion = platformVersion("build_tools")
    ndkVersion = platformVersion("ndk")

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }

    defaultConfig {
        minSdk = platformVersion("min_sdk").toInt()

        @Suppress("UnstableApiUsage")
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DZEPHR_NATIVE_DEPS_ROOT=$nativeDepsRootPath",
                    "-DZEPHR_NATIVE_DEPS_RECIPE_HASH=$nativeDepsRecipeHash"
                )
                cppFlags += listOf("-std=c++20", "-fexceptions", "-frtti")
            }
        }

        ndk {
            abiFilters += nativeAbis
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = platformVersion("cmake")
        }
    }
}

androidComponents {
    onVariants { variant ->
        val capitalized = variant.name.replaceFirstChar { it.uppercase() }
        val syncAssets = tasks.register<SyncAssetDirectory>("sync${capitalized}ModelManifestAssets") {
            sourceDir.set(rootProject.layout.projectDirectory.dir("../data/model_manifests"))
            outputDir.set(layout.buildDirectory.dir("generated/assets/modelManifests/${variant.name}"))
        }
        variant.sources.assets?.addGeneratedSourceDirectory(
            syncAssets,
            SyncAssetDirectory::outputDir,
        )

        val nativeConfig = if (variant.buildType == "release") "Release" else "Debug"
        val copyLiteRt = tasks.register<CopyLiteRtSharedLibs>("copy${capitalized}LiteRtSharedLibs") {
            nativeDepsRoot.set(nativeDepsRootPath)
            this.nativeConfig.set(nativeConfig)
            abis.set(nativeAbis)
            outputDir.set(layout.buildDirectory.dir("generated/jniLibs/${variant.name}"))
        }
        variant.sources.jniLibs?.addGeneratedSourceDirectory(
            copyLiteRt,
            CopyLiteRtSharedLibs::outputDir,
        )
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("org.jetbrains.kotlin:kotlin-reflect:2.3.21")
    implementation("io.modelcontextprotocol:kotlin-sdk-client:0.12.0")
    implementation("io.ktor:ktor-client-okhttp:3.3.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
    testImplementation("junit:junit:4.13.2")
}

publishing {
    repositories {
        maven {
            name = "PrivateArtifactRegistry"
            url = uri(sdkPrivateMavenUrl.get())
        }
    }

    publications {
        create<MavenPublication>("AarPrivate") {
            groupId = sdkGroupId.get()
            artifactId = sdkArtifactId.get()
            version = sdkVersion.get()

            afterEvaluate {
                from(components["release"])
            }

            pom {
                name.set("Zephr Agent Runtime")
                description.set("Android SDK wrapper for the Zephr Agent native runtime")
                url.set("https://github.com/zephr-xyz/diagnostics")

                licenses {
                    license {
                        name.set("Proprietary")
                        distribution.set("repo")
                    }
                }

                developers {
                    developer {
                        id.set("zephr-team")
                        name.set("Zephr Development Team")
                        email.set("support@zephr.xyz")
                    }
                }

                scm {
                    connection.set("scm:git:git://github.com/zephr-xyz/diagnostics.git")
                    developerConnection.set("scm:git:ssh://github.com/zephr-xyz/diagnostics.git")
                    url.set("https://github.com/zephr-xyz/diagnostics")
                }
            }
        }
    }
}

tasks.register("printZephrAgentRuntimePublicationCoordinates") {
    group = "publishing"
    description = "Prints the private Maven coordinates that will be published for the SDK AAR."

    doLast {
        println("${sdkGroupId.get()}:${sdkArtifactId.get()}:${sdkVersion.get()}")
        println("repository: ${sdkPrivateMavenUrl.get()}")
    }
}

tasks.register("publishZephrAgentRuntimeAar") {
    group = "publishing"
    description = "Builds and publishes the release Zephr Agent Runtime AAR to the private Maven repository."
    dependsOn("publishAarPrivatePublicationToPrivateArtifactRegistryRepository")
}
