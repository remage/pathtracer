-- premake5.lua
workspace "pathtracer"
	configurations { "debug", "profile", "release" }
	platforms { "x86", "x64" }
	defaultplatform "x64"
	targetdir "bin/%{cfg.buildcfg}"
	objdir "bin/obj"
	
	inlining "Explicit"
--	callingconvention "VectorCall"
	vectorextensions "SSE2"
	floatingpoint "Fast"

	floatingpointexceptions "Off"
	exceptionhandling "Off"
	flags {
		"NoBufferSecurityCheck",
		"NoImportLib",
	}

project "pathtracer"
	kind "ConsoleApp"
	language "C"
	cdialect "C11"
	toolset "clang"

	targetdir "bin/%{cfg.buildcfg}"

	files { 
		"source/*.h", 
		"source/*.c",
	}

	filter "configurations:debug"
		defines { "DEBUG", "_DEBUG" }
		symbols "On"

	filter "configurations:profile"
		defines { "NDEBUG", "PROFILE" }
		optimize "On"
		symbols "On"

		filter "configurations:release"
		defines { "NDEBUG", "RELEASE" }
		optimize "On"
