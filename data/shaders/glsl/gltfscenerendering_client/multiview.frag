#version 450

layout(set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout(set = 1, binding = 1) uniform sampler2D samplerNormalMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout(location = 5) in vec4 inTangent;

layout (location = 0) out vec4 outColor;

layout(constant_id = 0) const bool ALPHA_MASK = false;
layout(constant_id = 1) const float ALPHA_MASK_CUTOFF = 0.0f;

int CLIENTWIDTH = 1280;
int CLIENTHEIGHT = 720;
int FOVEAWIDTH = 320;
int FOVEAHEIGHT = 240;

void main() 
{
	vec4 color = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);

	int midpoint_of_eye_x = CLIENTWIDTH / 2;
	int midpoint_of_eye_y = CLIENTHEIGHT / 2;

	// Get the top left point for left eye
	int left_lefteye_x  = midpoint_of_eye_x - (FOVEAWIDTH / 2);
	int right_lefteye_x = midpoint_of_eye_x + (FOVEAWIDTH / 2);
	int top_eyepoint_y = midpoint_of_eye_y + (FOVEAHEIGHT / 4);
	int bottom_eyepoint_y = midpoint_of_eye_y - (FOVEAHEIGHT / 4);

	// Get the top left point for right eye -- y is same
	int left_righteye_x = (CLIENTWIDTH / 2) + midpoint_of_eye_x - (FOVEAWIDTH / 2);
	int right_righteye_x = (CLIENTWIDTH / 2) + midpoint_of_eye_x + (FOVEAWIDTH / 2);

	if((gl_FragCoord.x > left_lefteye_x && gl_FragCoord.x < right_lefteye_x && gl_FragCoord.y > bottom_eyepoint_y && gl_FragCoord.y < top_eyepoint_y))// ||
	//(gl_FragCoord.x > left_righteye_x && gl_FragCoord.x < right_righteye_x && gl_FragCoord.y > bottom_eyepoint_y && gl_FragCoord.y < top_eyepoint_y))
	//if(gl_FragCoord.x < midpoint_of_eye_x)
	{
		discard;
	}

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = cross(inNormal, inTangent.xyz) * inTangent.w;
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize(texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0));

	float ambient = 0.05;
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, N);
	vec3 diffuse = max(dot(N, L), ambient).rrr;
	float specular = 0.5 * clamp(pow(max(dot(R, V), 0.0), 16.0), 0.0, 0.25);
	outColor = vec4(diffuse * color.rgb + specular, color.a);
	//outColor = color;

}