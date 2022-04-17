#version 450

layout(set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout(set = 1, binding = 1) uniform sampler2D samplerNormalMap;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inViewVec;
layout(location = 4) in vec3 inLightVec;
layout(location = 5) in vec4 inTangent;
layout(location = 6) in vec3 inModelPos;

layout(location = 0) out vec4 outColor;

layout(constant_id = 0) const bool ALPHA_MASK = false;
layout(constant_id = 1) const float ALPHA_MASK_CUTOFF = 0.0f;


struct PointLight
{
	vec3 position;
	vec3 diffuse;
	vec3 specular;
};


vec3 point_light_diffuse(PointLight light, vec3 normal, vec3 model_position)
{
	float ambient = 0.05;
	float distance = length(light.position - model_position);
	vec3 light_direction = normalize(light.position - model_position);
	vec3 L = normalize(light_direction);
	vec3 diffuse = light.diffuse *  max(dot(L, normal), ambient).rrr * distance;
	float atten = 1.0 / (distance * distance);

	return atten * diffuse;
}


float point_light_specular(PointLight light, vec3 normal, vec3 model_position, vec3 view_direction)
{
	float ambient = 0.05;
	float distance = length(light.position - model_position);

	vec3 light_direction = normalize(light.position - model_position);
	vec3 R = reflect(-light_direction, normal);
	float spec = 0.5 * clamp(pow(max(dot(R, view_direction), 0.0), 16.0), 0.0, 0.25) * distance;
	float atten = 1.0 / (distance * distance);

	return atten * spec;
}



void main() 
{
	vec4 color = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);

	PointLight pointlights[4];
	pointlights[0].position = vec3(-8.2, 0, 5.25);
	pointlights[1].position = vec3(-6.2, 0.25, 5.25);
	pointlights[2].position = vec3(-4.2, 0.5, 5.25);
	pointlights[3].position = vec3(-2.2, 0.75, 5.25);

	pointlights[0].diffuse = vec3(1.0, 0.0, 1.0);
	pointlights[1].diffuse = vec3(0.0, 1.0, 1.0);
	pointlights[2].diffuse = vec3(1.0, 1.0, 0.0);
	pointlights[3].diffuse = vec3(0.4, 0.2, 0.6);

	pointlights[0].specular = vec3(0.2, 0.2, 0.2);
	pointlights[1].specular = vec3(0.4, 0.1, 0.7);
	pointlights[2].specular = vec3(0.6, 0.6, 0.1);
	pointlights[3].specular = vec3(0.5, 0.1, 0.4);


	if(ALPHA_MASK)
	{
		if(color.a < ALPHA_MASK_CUTOFF)
		{
			discard;
		}
	}

	
	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize(texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0));

	vec3 diffuse = vec3(0.0, 0.0, 0.0);
	float spec = 0.0;

	for(int i = 0; i < 4; i++)
	{
		diffuse += point_light_diffuse(pointlights[i], N, inModelPos);
		spec += point_light_specular(pointlights[i], N, inModelPos, inViewVec);
	}


	outColor = vec4(diffuse * color.rgb + spec, color.a);
	//outColor = color;

}