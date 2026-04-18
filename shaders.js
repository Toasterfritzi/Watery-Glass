// ============================================================
// shaders.js — All GLSL shader sources for the fluid simulation
// ============================================================

// --- Shared vertex shader (fullscreen quad) ---
export const baseVertexShader = `
  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform vec2 texelSize;
  void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`;

// --- Splat: inject velocity/dye into the simulation ---
export const splatShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec3 color;
  uniform vec2 point;
  uniform float radius;
  void main () {
    vec2 p = vUv - point;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4(base + splat, 1.0);
  }
`;

// --- Advection: move the fluid field along its own velocity ---
export const advectionShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uVelocity;
  uniform sampler2D uSource;
  uniform vec2 texelSize;
  uniform float dt;
  uniform float dissipation;
  void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    vec4 result = dissipation * texture2D(uSource, coord);
    gl_FragColor = result;
  }
`;

// --- Divergence of the velocity field ---
export const divergenceShader = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`;

// --- Pressure solve (Jacobi iteration) ---
export const pressureShader = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uDivergence;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    float div = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - div) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
  }
`;

// --- Gradient subtraction: make velocity divergence-free ---
export const gradientSubtractShader = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

// --- Curl for vorticity ---
export const curlShader = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }
`;

// --- Vorticity confinement force ---
export const vorticityShader = `
  precision highp float;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  uniform sampler2D uCurl;
  uniform float curl;
  uniform float dt;
  void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity += force * dt;
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

// --- Clear shader ---
export const clearShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform float value;
  void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
  }
`;

// --- Display: render the fluid as a 3D-looking water surface ---
export const displayShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform sampler2D uVelocity;
  uniform vec2 texelSize;
  uniform float time;

  // Water color palette
  vec3 deepColor   = vec3(0.02, 0.07, 0.18);
  vec3 midColor    = vec3(0.05, 0.20, 0.45);
  vec3 surfColor   = vec3(0.15, 0.55, 0.85);
  vec3 foamColor   = vec3(0.70, 0.90, 0.98);
  vec3 causticColor = vec3(0.30, 0.85, 0.95);

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 5; i++) {
      v += a * noise(p);
      p = rot * p * 2.0;
      a *= 0.5;
    }
    return v;
  }

  void main () {
    // Sample velocity for distortion
    vec2 vel = texture2D(uVelocity, vUv).xy;
    float speed = length(vel) * 0.0005;

    // Create normal map from velocity neighbors
    float hL = length(texture2D(uVelocity, vUv - vec2(texelSize.x, 0.0)).xy);
    float hR = length(texture2D(uVelocity, vUv + vec2(texelSize.x, 0.0)).xy);
    float hU = length(texture2D(uVelocity, vUv + vec2(0.0, texelSize.y)).xy);
    float hD = length(texture2D(uVelocity, vUv - vec2(0.0, texelSize.y)).xy);
    vec3 normal = normalize(vec3(hL - hR, hD - hU, 0.3));

    // Animated procedural waves
    vec2 waveUv = vUv * 6.0 + time * 0.05;
    float wave1 = fbm(waveUv + vel * 0.002);
    float wave2 = fbm(waveUv * 1.5 - time * 0.08);
    float waves = (wave1 + wave2) * 0.5;

    // Fresnel-like rim lighting
    float fresnel = pow(1.0 - abs(normal.z), 3.0);

    // Combine height from velocity + waves
    float height = speed * 50.0 + waves * 0.3;

    // Water color based on depth/height
    vec3 waterColor = mix(deepColor, midColor, smoothstep(0.0, 0.3, height));
    waterColor = mix(waterColor, surfColor, smoothstep(0.3, 0.6, height));

    // Caustics: bright dancing light patterns
    vec2 causticUv = vUv * 12.0 + time * 0.12 + vel * 0.003;
    float c1 = noise(causticUv);
    float c2 = noise(causticUv * 1.7 + 3.14);
    float caustic = pow(smoothstep(0.3, 0.7, c1 * c2 * 2.5), 2.0);
    waterColor += causticColor * caustic * 0.35;

    // Specular highlights
    vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
    float spec = pow(max(dot(normal, lightDir), 0.0), 64.0);
    waterColor += vec3(1.0) * spec * 0.6;

    // Foam on high-velocity areas
    float foam = smoothstep(0.012, 0.04, speed);
    waterColor = mix(waterColor, foamColor, foam * 0.5);

    // Fresnel edge glow
    waterColor += causticColor * fresnel * 0.25;

    // Subtle refraction distortion from dye texture
    vec2 refractUv = vUv + normal.xy * 0.02;
    vec3 dye = texture2D(uTexture, refractUv).rgb;
    waterColor += dye * 0.15;

    // Vignette
    float vig = 1.0 - 0.3 * length(vUv - 0.5);

    gl_FragColor = vec4(waterColor * vig, 1.0);
  }
`;
