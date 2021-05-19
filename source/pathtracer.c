
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <intrin.h>
#include <immintrin.h>
#include <x86intrin.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#if !__has_attribute(ext_vector_type)
#error Unsupported C language extension 'ext_vector_type', use Clang to compile.
#endif

typedef float __attribute__((ext_vector_type(3))) float3;
typedef float __attribute__((ext_vector_type(4))) float4;

inline float __vectorcall dot(float3 v1, float3 v2) { float3 v = v1 * v2; return v.x + v.y + v.z; }
inline float3 __vectorcall normalize(float3 v) { return v / sqrtf(dot(v, v)); }

inline float3 __vectorcall lerp(float3 v1, float3 v2, float3 t) { return v1 + (v2 - v1) * t; }

// #fixme MSVC has _mm_pow_ps(), an SSE parallel pow implementation.
inline float3 __vectorcall f3_powf(float3 v1, float f) { return (float3){ powf(v1.x, f), powf(v1.y, f), powf(v1.z, f) }; }
#define pow(a, b) _Generic((a), float3:f3_powf, float:powf, default:pow)(a, b)


// Fast signed floating-point [-1.0, +1.0] random generator.
// https://www.iquilezles.org/www/articles/sfrand/sfrand.htm
inline float sfrand()
{
	// #fixme thread_local?
	static unsigned int seed = 303;
	float res;
	seed *= 16807;
	*((unsigned int *) &res) = (seed>>9) | 0x40000000;
	return res - 3.0f;
}

// Generate a random point on the surface of a unit sphere.
float3 __vectorcall f3_rand_unit_sphere()
{
	float z = sfrand();
	float a = sfrand() * M_PI;
	float r = sqrtf(1.0f - z*z);
	float x = r * cosf(a);
	float y = r * sinf(a);
	return (float3){ x, y, z };
}

float3 __vectorcall srgb_to_linear(float3 color)
{
	// (color <= 0.04045) ? (color / 12.92) : pow((color + 0.055) / 1.055, 2.4);
	float4 mask = _mm_cmple_ps(color.xyzz, (float4)(0.04045f));
	return lerp(pow((color + 0.055) / 1.055, 2.4), color / 12.92, mask.xyz);
}

// Approximation for linear to sRGB transformation.
// http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
float3 __vectorcall linear_to_srgb(float3 color)
{
	float4 c1 = _mm_sqrt_ps(color.xyzz);
	float4 c2 = _mm_sqrt_ps(c1);
	float4 c3 = _mm_sqrt_ps(c2);
//	return max(0, 0.585122381 * c1 + 0.783140355 * c2 - 0.368262736 * c3).xyz;
	return max(0, 0.662002687 * c1 + 0.684122060 * c2 - 0.323583601 * c3 - 0.0225411470 * color.xyzz).xyz;
//	return max(0, 1.055 * pow(color, 0.416666667) - 0.055);
}


typedef struct pt_ray pt_ray;
struct pt_ray
{
	float3 origin, dir;
};

inline float3 __vectorcall pt_ray_trace(const pt_ray ray, float t)
{
	return ray.origin + ray.dir * t;
}

typedef struct pt_hit pt_hit;
struct pt_hit
{
	float3 point;
	float3 normal;
	float t;
	int id;
};

typedef struct pt_material pt_material;
struct pt_material
{
	float3 color;
};

struct pt_geom;
typedef bool pt_trace_fn(const struct pt_geom* geom, pt_hit* hit, const pt_ray ray, float t_min, float t_max);
typedef bool pt_scatter_fn(const struct pt_geom* geom, const struct pt_material* mat, const pt_hit* hit, pt_ray* ray, float3* attn);


typedef struct pt_geom_sphere pt_geom_sphere;
struct pt_geom_sphere
{
	float3 center;
	float radius;
};

typedef struct pt_geom_plane pt_geom_plane;
struct pt_geom_plane
{
	float3 p0;
	float3 normal;
};

typedef struct pt_geom pt_geom;
struct pt_geom
{
	pt_trace_fn* trace_fn;
	pt_scatter_fn* scatter_fn;
	union
	{
		pt_geom_plane plane;
		pt_geom_sphere sphere;
	};
	pt_material mat;
};

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

// Geom: Sky
bool pt_geom_sky_trace(const pt_geom* geom, pt_hit* hit, const pt_ray ray, float t_min, float t_max)
{
	if (t_max == FLT_MAX)
	{
		hit->t = t_max;
		hit->point = pt_ray_trace(ray, t_max);
		hit->normal = -ray.dir;
		return true;
	}
	return false;
}

// Scene
bool pt_scene_trace(const pt_geom* geoms, pt_hit* hit, const pt_ray ray, float t_min, float t_max)
{
	pt_hit temp = { .t = t_max, .id = -1 };
	for (int i = 0; geoms[i].trace_fn != NULL; i++)
	{
		if (geoms[i].trace_fn(&geoms[i], &temp, ray, t_min, temp.t))
		{
			temp.id = i;
			*hit = temp;
		}
	}
	return temp.id >= 0;
}

bool pt_scene_scatter(const pt_geom* geoms, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	return geoms[hit->id].scatter_fn(&geoms[hit->id], &geoms[hit->id].mat, hit, ray, attn);
}

// BRDF: Diffuse.
bool pt_scatter_diffuse(const struct pt_geom* geom, const pt_material* mat, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	// Lambertian
	*attn *= mat->color;

	// Next ray; cosine-weighted distribution fn.
	ray->origin = hit->point;
	ray->dir = normalize(hit->normal + f3_rand_unit_sphere());
/*
	// Next ray; uniform distribution fn, separate cosine-weighting.
	ray->dir = normalize(f3_rand_unit_sphere());
	if (dot(ray->dir, hit->normal) < 0)
		ray->dir *= -1.0f;
	*attn *= 2.0 * dot(ray->dir, hit->normal);
*/
	return true;
}

bool pt_scatter_oren_nayar(const struct pt_geom* geom, const pt_material* mat, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	// Reflect the incoming ray to a random direction (cosine-weighted distribution fn).
	float3 refl = normalize(hit->normal + f3_rand_unit_sphere());

	float roughness = 0.5;
	float sigma = roughness / sqrt(2.0);
	float sigma2 = sigma*sigma;

	float n_dot_l = dot(hit->normal, refl);
	float n_dot_v = dot(hit->normal, -ray->dir);
	float l_dot_v = dot(refl, -ray->dir);
	float s = l_dot_v - n_dot_l * n_dot_v;
	float t = s <= 0.0 ? 1.0 : max(n_dot_l, n_dot_v);
	float3 a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33) + 0.17 * mat->color * sigma2 / (sigma2 + 0.13);
	float b = 0.45 * sigma2 / (sigma2 + 0.09);

	*attn *= mat->color * (a + b * s/t);

	// Next ray; 
	ray->origin = hit->point;
	ray->dir = refl;
	return true;
}

// #todo BRDF: Light source.
bool pt_scatter_light(const pt_geom* geom, const pt_material* mat, const pt_hit* hit, pt_ray* ray, float3* attn)
{
	*attn *= mat->color;
	return false;
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

void progress(int y)
{
	if (((y+1) % (image_height/10)) == 0)
		printf(".");
}

void render_sphere3_v1()
{
	uint8_t* image_data_srgb = (uint8_t*)malloc(sizeof(uint8_t) * 3 * image_width * image_height);
	memset(image_data_srgb, 0, sizeof(uint8_t) * 3 * image_width * image_height);

	// Scene definition.
	pt_geom scene[] = 
	{ 
		{ &pt_geom_plane_trace,  &pt_scatter_diffuse, .plane  = { (float3){ 0.0f, -1.0f, 0.0f }, (float3){ 0.0f, 1.0f, 0.0f }}, .mat = { (float3){ 0.5f, 0.5f, 0.5f }}},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){  0.0f, 0.0f, 0.0f }, 1.0f },						.mat = { srgb_to_linear((float3){ 0.788f, 0.122f, 0.216f }) }},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){ -2.0f, 0.0f, 0.0f }, 1.0f },						.mat = { (float3){ 1.0f, 1.0f, 1.0f }}},
//		{ &pt_geom_sphere_trace, &pt_scatter_oren_nayar, .sphere = { (float3){  2.0f, 0.0f, 0.0f }, 1.0f },						.mat = { (float3){ 0.3f, 0.3f, 0.3f }}},
		{ &pt_geom_sphere_trace, &pt_scatter_diffuse, .sphere = { (float3){  2.0f, 0.0f, 0.0f }, 1.0f },						.mat = { (float3){ 0.3f, 0.3f, 0.3f }}},
//		{ &pt_geom_sphere_trace, &pt_scatter_light,   .sphere = { (float3){  1.0f, 1.732f, 0.0f }, 1.0f },                      .mat = { (float3){ 1.0f, 1.0f, 1.0f }}},
		{ &pt_geom_sky_trace,    &pt_scatter_light,   																			.mat = { (float3){ 0.5f, 0.5f, 0.5f }}},
		{ NULL }
	};

	// Camera setup.
	float3 cam_origin = { 0.0f, 0.0f, 5.0f };
	float3 cam_target = { 0.0f, 0.0f, 0.0f };
	float3 cam_dir = normalize(cam_target - cam_origin);

	// #todo Calculate proper view matrix from cam_origin, cam_target and up vector.
	float3 cam_h = { 1.0f, 0.0f, 0.0f }; // fov_h = 90
	float3 cam_v = { 0.0f, - (float)image_height / (float)image_width, 0.0f };

	uint64_t tsc0 = __rdtsc();

	// Path tracing!
	int i = 0;
	for (int y = 0; y < image_height; y++)
	{
		progress(y);
		for (int x = 0; x < image_width; x++)
		{
			float3 color_acc = { 0.0f, 0.0f, 0.0f };
			for (int s = 0; s < num_samples; s++)
			{
				// #todo Gaussian AA, instead of box.
				float aa_x = 0.5f * sfrand();
				float aa_y = 0.5f * sfrand();
				float ray_x = ((float)x + aa_x) / (float)(image_width-1) * 2.0f - 1.0f;
				float ray_y = ((float)y + aa_y) / (float)(image_height-1) * 2.0f - 1.0f;

				float3 ray_dir = cam_dir + ray_x * cam_h + ray_y * cam_v;
				pt_ray ray = { cam_origin, normalize(ray_dir) };

				float3 attn = { 1.0f, 1.0f, 1.0f };
				for (int b = 0; b < num_bounces; b++)
				{
					pt_hit hit;
					if (!pt_scene_trace(scene, &hit, ray, 0.0001f, FLT_MAX))
					{
						// No hit, terminate ray.
						break; 
					}
					if (!pt_scene_scatter(scene, &hit, &ray, &attn))
					{
						// Light hit: accumulate color and terminate ray.
						color_acc += attn;
						break;
					}
				}
			}

			color_acc /= (float)num_samples;

			// Saturation
			color_acc = min(max(color_acc, 0.0f), 1.0f);

			// Gamma correction
			color_acc = linear_to_srgb(color_acc);

			// #todo Reconstruction filters.
			image_data_srgb[i++] = (int)(255 * color_acc.x);
			image_data_srgb[i++] = (int)(255 * color_acc.y);
			image_data_srgb[i++] = (int)(255 * color_acc.z);
		}
	}

	uint64_t tsc = __rdtsc() - tsc0;
	printf(" CPU cycles = %llu M\n", tsc / 1000000);

	stbi_write_png("sphere3_v1.png", image_width, image_height, 3, image_data_srgb, image_width * 3);
}

int __cdecl main(int argc, char *argv[])
{
	render_sphere3_v1();
}
