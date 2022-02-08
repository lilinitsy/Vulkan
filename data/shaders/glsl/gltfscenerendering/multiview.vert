#version 450

#extension GL_EXT_multiview : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inTangent;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec3 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;
layout (location = 5) out vec4 outTangent;


layout (set = 0, binding = 0) uniform UBO 
{
	mat4 projection[2];
	mat4 view[2];
	vec4 lightPos;
	//vec4 viewpos;
} ubo;

layout(push_constant) uniform PushConsts
{
	mat4 model;
} primitive;

void main() 
{
	outColor = inColor;
	outUV = inUV;
	outNormal = mat3(ubo.view[gl_ViewIndex] * primitive.model) * inNormal;

	vec4 pos = vec4(inPos.xyz, 1.0);
	vec4 worldPos = (ubo.view[gl_ViewIndex] * primitive.model) * pos;
		
	vec3 lPos = vec3((ubo.view[gl_ViewIndex] * primitive.model) * ubo.lightPos);
	outLightVec = lPos - worldPos.xyz;
	outViewVec = -worldPos.xyz;	
	//outViewVec = ubo.viewpos.xyz - pos.xyz;
	outTangent = inTangent;
	

	gl_Position = ubo.projection[gl_ViewIndex] * worldPos;
}
