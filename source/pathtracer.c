
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#define WIN32_LEAN_AND_MEAN
#define MICROSOFT_WINDOWS_WINBASE_H_DEFINE_INTERLOCKED_CPLUSPLUS_OVERLOADS 0
#include <Windows.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef nullptr
#define nullptr (void*)0
#endif


#if !__has_attribute(ext_vector_type)
#error Unsupported C language extension 'ext_vector_type'.
#endif

typedef float __attribute__((ext_vector_type(3))) float3;
typedef float __attribute__((ext_vector_type(4))) float4;

inline float __vectorcall dot(float3 v1, float3 v2) { float3 v = v1 * v2; return v.x + v.y + v.z; }
inline float3 __vectorcall normalize(float3 v) { return v / sqrtf(dot(v, v)); }

inline float3 __vectorcall lerp(float3 v1, float3 v2, float3 t) { return v1 + (v2 - v1) * t; }

// #fixme MSVC has _mm_pow_ps(), an SSE parallel pow implementation.
inline float3 __vectorcall f3_powf(float3 v1, float f) { return (float3){ powf(v1.x, f), powf(v1.y, f), powf(v1.z, f) }; }
#define pow(a, b) _Generic((a), float3:f3_powf, float:powf, default:pow)(a, b)


float frand()
{
	return (float)rand() / (float)RAND_MAX;
}

// Generate a random point on the surface of a unit sphere.
float3 f3_rand_unit_sphere()
{
	float z = frand() * 2.0f - 1.0f;
	float a = frand() * 2.0f * (float)M_PI;
	float r = sqrtf(1.0f - z*z);
	float x = r * cosf(a);
	float y = r * sinf(a);
	return (float3){ x, y, z };
}

float3 srgb_to_linear(float3 color)
{
	// (color <= 0.04045) ? (color / 12.92) : pow((color + 0.055) / 1.055, 2.4);
	float4 mask = _mm_cmple_ps(color.xyzz, (float4)(0.04045f));
	return lerp(pow((color + 0.055) / 1.055, 2.4), color / 12.92, mask.xyz);
}

float3 linear_to_srgb(float3 color)
{
	// max(0, 1.055 * pow(color, 0.416666667) - 0.055);
	return max(0, 1.055 * pow(color, 0.416666667) - 0.055);
}


struct pt_ray
{
	float3 origin, dir;
};
typedef struct pt_ray pt_ray;

float3 pt_ray_trace(const pt_ray ray, float t)
{
	return ray.origin + ray.dir * t;
}

struct pt_hit
{
	float3 point;
	float3 normal;
	float t;
	int id;
};
typedef struct pt_hit pt_hit;

struct pt_geom;
typedef bool pt_trace_fn(const struct pt_geom* geom, pt_hit* hit, const pt_ray ray, float t_min, float t_max);
typedef bool pt_scatter_fn(const struct pt_geom* geom, const pt_hit* hit, pt_ray* ray, float3* attn);


struct pt_geom_sphere
{
	float3 center;
	float radius;
};
typedef struct pt_geom_sphere pt_geom_sphere;

struct pt_geom_plane
{
	float3 p0;
	float3 normal;
};
typedef struct pt_geom_plane pt_geom_plane;

struct pt_geom
{
	pt_trace_fn* trace_fn;
	pt_scatter_fn* scatter_fn;
	union
	{
		pt_geom_plane plane;
		pt_geom_sphere sphere;
	};
};
typedef struct pt_geom pt_geom;

// Geom: Sphere
bool pt_geom_sphere_trace(const pt_geom* geom, pt_hit* hit, const pt_ray ray, float t_min, float t_max)
{
	assert(geom->trace_fn == &pt_geom_sphere_trace);
	pt_geom_sphere sph = geom->sphere;

	float3 oc = ray.origin - sph.center;
	float a = dot(ray.dir, ray.dir);
	float b = 2.0f * dot(oc, ray.dir);
	float c = dot(oc, oc) - sph.radius*sph.radius;
	float d = b*b - 4.0f*a*c;
	if (d >= 0)
	{
		float t = (-b - sqrtf(d)) / (2.0f*a);
		if (t >= t_min && t <= t_max)
		{
			hit->t = t;
			hit->point = pt_ray_trace(ray, t);
			hit->normal = (hit->point - sph.center) / sph.radius;
			return true;
		}
		t = (-b + sqrtf(d)) / (2.0f*a);
		if (t >= t_min && t <= t_max)
		{
			hit->t = t;
			hit->point = pt_ray_trace(ray, t);
			hit->normal = (hit->point - sph.center) / sph.radius;
			return true;
		}
	}
	return false;
}

// Geom: Plane
bool pt_geom_plane_trace(const pt_geom* geom, pt_hit* hit, const pt_ray ray, float t_min, float t_max)
{
	assert(geom->trace_fn == &pt_geom_plane_trace);
	pt_geom_plane pl = geom->plane;

	const float epsilon = 1e-6f;
	float d = dot(pl.normal, ray.dir);
	if (fabsf(d) > epsilon)
	{
		float3 op0 = pl.p0 - ray.origin;
		float t = dot(op0, pl.normal) / d;
		if (t >= t_min && t <= t_max)
		{
			hit->t = t;
			hit->point = pt_ray_trace(ray, t);
			hit->normal = pl.normal;
			return true;
		}
	}
	return false;
}

// Scene
bool pt_scene_trace(const pt_geom* geoms, pt_hit* hit, const pt_ray ray, float t_min, float t_max)
{
	pt_hit temp = { .t = t_max, .id = -1 };
	for (int i = 0; geoms[i].trace_fn != nullptr; i++)
	{
		if (geoms[i].trace_fn(&geoms[i], &temp, ray, t_min, temp.t))
		{
			temp.id = i;
			*hit = temp;
		}
	}
	return temp.id >= 0;
}

bool pt_scene_scatter(const struct pt_geom* geoms, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	return geoms[hit->id].scatter_fn(&geoms[hit->id], hit, ray, attn);
}

// BRDF: Diffuse.
bool pt_scatter_diffuse(const struct pt_geom* geom, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	const float3 color_gray3 = { 0.3f, 0.3f, 0.3f };
	const float3 color_gray5 = { 0.5f, 0.5f, 0.5f };
	const float3 color_white = { 1.0f, 1.0f, 1.0f };
	const float3 color_crimson = srgb_to_linear((float3){ 0.788f, 0.122f, 0.216f }); // 201,31,55 Karakurenai, foreign crimson

	// #fixme Material color needs to go to the scene definition.
	float3 materials[] = 
	{
		color_gray5,
		color_crimson,
		color_white,
		color_gray3
	};

	// Lambertian
	*attn = (*attn) * materials[hit->id];

	// Next ray; cosine-weighted distribution fn.
	ray->origin = hit->point;
	ray->dir = normalize(hit->normal + f3_rand_unit_sphere());
/*
	// Next ray; uniform distribution fn, separate cosine-weighting.
	ray->dir = normalize(f3_rand_unit_sphere());
	if (dot(ray->dir, hit->normal) < 0)
		ray->dir *= -1.0f;
	*attn *= dot(ray->dir, hit->normal) * 2;
*/
	return true;
}

// #todo BRDF: Light source. 
bool pt_scatter_light(const struct pt_geom* geom, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	const float3 color_white = (float3){ 1.0f, 1.0f, 1.0f };
	*attn = (*attn) * color_white;
	return false;
}

// Background
float3 pt_background(pt_ray ray)
{
	const float3 color_gray5 = (float3){ 0.5f, 0.5f, 0.5f };
	return color_gray5;
/*
	float3 dir = normalize(ray.dir);
	float t = 0.5f * (dir.y + 1.0f);
	const float3 color_gray3 = (float3){ 0.3f, 0.3f, 0.3f };
	const float3 color_skyblue = srgb_to_linear((float3){ 0.3f, 0.56f, 0.675f }); // 77,143,172 Sora-iro, sky blue
	return lerp(color_gray3, color_skyblue, t*t);
*/
}


const int image_width = 1280;
const int image_height = 720;
const int num_samples = 256;
const int num_bounces = 8;

void progressbar(int y)
{
	if (((y+1) % (image_height/10)) == 0)
	{
		OutputDebugStringA(".");
		printf(".");
	}
}

void render_sphere3_v1()
{
	uint8_t* image_data_srgb = (uint8_t*)malloc(sizeof(uint8_t) * 3 * image_width * image_height);
	memset(image_data_srgb, 0, sizeof(uint8_t) * 3 * image_width * image_height);

	// Scene definition.
	pt_geom scene[] = 
	{ 
		{ &pt_geom_plane_trace,  &pt_scatter_diffuse, .plane  = { (float3){ 0.0f, -1.0f, 0.0f }, (float3){ 0.0f, 1.0f, 0.0f }}},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){  0.0f, 0.0f, 0.0f }, 1.0f }},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){ -2.0f, 0.0f, 0.0f }, 1.0f }},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){  2.0f, 0.0f, 0.0f }, 1.0f }},
		{ nullptr }
	};

	// Camera setup.
	float3 cam_origin = { 0.0f, 0.0f, 5.0f };
	float3 cam_target = { 0.0f, 0.0f, 0.0f };
	float3 cam_dir = normalize(cam_target - cam_origin);

	// #todo Calculate proper view matrix from cam_origin, cam_target and up vector.
	float3 cam_h = { 1.0f, 0.0f, 0.0f }; // fov_h = 90
	float3 cam_v = { 0.0f, - (float)image_height / (float)image_width, 0.0f };

	// Path tracing!
	int i = 0;
	for (int y = 0; y < image_height; y++)
	{
		progressbar(y);
		for (int x = 0; x < image_width; x++)
		{
			float3 color_acc = { 0.0f, 0.0f, 0.0f };
			for (int s = 0; s < num_samples; s++)
			{
				// #todo Gaussian AA, instead of box.
				float aa_x = (float)rand() / (float)(RAND_MAX) - 0.5f;
				float aa_y = (float)rand() / (float)(RAND_MAX) - 0.5f;
				float ray_x = ((float)x + aa_x) / (float)(image_width-1) * 2.0f - 1.0f;
				float ray_y = ((float)y + aa_y) / (float)(image_height-1) * 2.0f - 1.0f;

				float3 ray_dir = cam_dir + ray_x * cam_h + ray_y * cam_v;
				pt_ray ray = { cam_origin, normalize(ray_dir) };

				float3 attn = { 1.0f, 1.0f, 1.0f };
				for (int b = 0; b < num_bounces; b++)
				{
					pt_hit hit;
					if (!pt_scene_trace(scene, &hit, ray, 0.0001f, FLT_MAX) ||
						!pt_scene_scatter(scene, &hit, &ray, &attn))
					{
						// No hit: background color, terminate ray.
						color_acc += attn * pt_background(ray);
						break;
					}
				}
			}

			color_acc /= (float)num_samples;

			// Saturation
			color_acc = min(max(color_acc, 0.0f), 1.0f);

			// Gamma correction
			color_acc = linear_to_srgb(color_acc);

			image_data_srgb[i++] = (int)(255 * color_acc.x);
			image_data_srgb[i++] = (int)(255 * color_acc.y);
			image_data_srgb[i++] = (int)(255 * color_acc.z);
		}
	}

	stbi_write_png("sphere3_v1.png", image_width, image_height, 3, image_data_srgb, image_width * 3);
}

int __cdecl main(int argc, char *argv[])
{
	render_sphere3_v1();
}
