precision mediump float;
uniform float time;
uniform sampler2D backbuffer;
uniform sampler2D camera;
uniform sampler2D samples;
uniform sampler2D spectrum;
uniform float volume;


uniform vec2 resolution;

const float tFrag = 1.0 / 512.0;
const float nFrag = 1.0 / 64.0;

vec2 rotate(in vec2 p, in float t) {
  return mat2(
    sin(t), cos(t),
    cos(t), -sin(t)
  ) * p;
}

float rand1(vec2 co){

    return 2.*fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453)-1.;
}

float rand2(vec2 uv, float p){
  return fract(sin(uv.x)*tan(uv.x)*cos(uv.x)*pow(uv.x,2.0));
}

vec3 mod289(vec3 x){
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x){
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x){
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r){
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v){
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

vec3 snoiseVec3( vec3 x ){

	float s  = snoise(vec3( x ));
	float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
	float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
	vec3 c = vec3( s , s1 , s2 );
	return c;

}

float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitud = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitud * noise(st);
        st *= 2.;
        amplitud *= .5;
    }
    return value;
}

// ray march
vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));

float dist_func(vec3 pos, float size)
{
    return length(pos) - size;
}

vec3 getNormal(vec3 pos, float size)
{
    float ep = 0.0001;
    return normalize(vec3(
            dist_func(pos, size) - dist_func(vec3(pos.x - ep, pos.y, pos.z), size),
            dist_func(pos, size) - dist_func(vec3(pos.x, pos.y - ep, pos.z), size),
            dist_func(pos, size) - dist_func(vec3(pos.x, pos.y, pos.z - ep), size)
        ));
}

vec4 qmult(vec4 a, vec4 b) {
	vec4 r;
	r.x = a.x * b.x - dot(a.yzw, b.yzw);
	r.yzw = a.x * b.yzw + b.x * a.yzw + cross(a.yzw, b.yzw);
	return r;
}

void julia(inout vec4 z, inout vec4 dz, in vec4 c) {
	for(int i = 0; i < 10; i++) {
		dz = 2.0 * qmult(z, dz);
		z = qmult(z, z) + c;

		if(dot(z, z) > 3.0) {
			break;
		}
	}
}

vec3 transform(vec3 p) {
	float t = time * .03;
	p.xy *= mat2(cos(t), sin(t), -sin(t), cos(t));
	t = time * .07;
	p.zx *= mat2(cos(t), sin(t), -sin(t), cos(t));
	return p;
}

float dist(in vec3 p, float x, float y) {
	p = transform(p);
	vec4 z = vec4(p, 0.0);
	vec4 dz = vec4(1.0, 0.0, 0.0, 0.0);

	vec2 m = vec2(x, y);
	vec4 c = vec4(m.x, m.y, 0.0, 0.0);

	julia(z, dz, c);

	float lz = length(z);
	float d = 0.5 * log(lz) * lz / length(dz) ;

	return d;
}


vec4 getTex(sampler2D buf){
  vec2 texPos = vec2(gl_FragCoord.xy/resolution);
  return texture2D(buf, texPos);
}

void main(){

    float col = 0.0;
    vec3 fcol = vec3(col);

    // centering
    vec2 uv = 2.0 * (gl_FragCoord.xy / resolution) - 1.0;

    // 1
    //col  = abs(1.0 / (uv.y) * 0.01);
    //fcol = vec3(col);

    // 2
    //col = abs(1.0 / (uv.y) * tan(time * 0.5) * 0.01);
    //fcol = vec3(col);

    // 3
    //col = abs(1.0 / (uv.y) * tan(time * 0.5 * uv.y) * 0.01);
    //fcol = vec3(col);

    // 4
    //uv = updateUV1(uv, 2.0);
    //col = abs(1.0 / (uv.y) * tan(time * 0.5 * uv.y) * 0.01);
    //fcol = vec3(col);

    // 5
    //col = sin((20.0 * uv.x) - 1.0);
    //col *= 0.1 * volume;
    //fcol += vec3(col);

    // 6
    //vec2 uv2 = uv;
    //uv2.x += time * 0.1;
    //col = sin((20.0 * uv2.x) - 1.0);
    //col *= 0.05 * volume;
    //fcol += vec3(col);

    // 7
    // delete horizon line
    //col = abs(1.0 / (uv.y) * tan(time * 0.5 * uv.y) * 0.01);
    //fcol += vec3( col*sin(time/2.)*0.2, col*cos(time) , 3.*sin(col + time / 3.0) * 0.75);

    // 8
    //uv = rotate(uv, time);
    //uv = mod(uv, vec2(3.0)) - vec2(1.0);
    //float v0 = 6.0;
    //float v1 = 40.0 * snoise(vec3(floor(uv.x * v0)/v0, floor(uv.y * v0)/v0,time*0.8));
    //vec3 vv = snoiseVec3(vec3(floor(uv.x * v1)/v1, floor(uv.y * v1)/v1, time*0.4));
    //fcol += vec3(vv);
    //fcol += vec3(vv * volume * 0.1);

    float v0 = 6.0;
    float v1 = 40.0 * snoise(vec3(floor(uv.x * v0)/v0, floor(uv.y * v0)/v0,time*0.8));
    vec3 vv = snoiseVec3(vec3(floor(uv.x * v1)/v1, floor(uv.y * v1)/v1, time*0.4));
    fcol += vec3(vv);
    fcol += vec3(vv * volume * 0.1);




    gl_FragColor = vec4(fcol, 1.0);
}
