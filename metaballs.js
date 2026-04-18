// ============================================================
// metaballs.js — WebGL Metaball Renderer for Glass UI Bubbles
// Renders liquid glass blobs that merge/split at element positions
// ============================================================

const VERT = `
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
varying vec2 vUv;
uniform vec2 uResolution;
uniform float uTime;
uniform int uCount;
uniform vec4 uBlobs[24];
uniform float uRadii[24];
uniform float uSpread;
uniform float uThreshold;
uniform vec3 uCursor;  // xy = position in canvas px (Y-flipped), z = radius
uniform float uCursorActive; // 1.0 when cursor is near an element, 0.0 otherwise

float sdRoundBox(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float fieldBlobs(vec2 p) {
  float f = 0.0;
  for (int i = 0; i < 24; i++) {
    if (i >= uCount) break;
    vec2 c = uBlobs[i].xy;
    vec2 hs = uBlobs[i].zw;
    float r = uRadii[i];
    float sdf = sdRoundBox(p - c, hs, r);
    float d = max(sdf, 0.0);
    f += uSpread * uSpread / (d * d + uSpread * uSpread);
  }
  return f;
}

float field(vec2 p) {
  float f = fieldBlobs(p);
  // Cursor blob — only adds contribution when near existing blobs (bridges, never standalone)
  if (uCursorActive > 0.5 && uCursor.z > 0.0) {
    float cd = length(p - uCursor.xy) - uCursor.z;
    cd = max(cd, 0.0);
    float cursorSpread = uSpread * 2.2;
    float cursorField = cursorSpread * cursorSpread / (cd * cd + cursorSpread * cursorSpread);
    // Only contribute where there's already some blob field nearby
    // This prevents the cursor from being a visible standalone bubble
    float nearBlob = smoothstep(0.05, 0.35, f);
    f += cursorField * nearBlob;
  }
  return f;
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), f.x),
             mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
}

void main() {
  vec2 px = vUv * uResolution;
  float f = field(px);

  if (f < uThreshold - 0.05) { discard; }

  // Surface edge
  float edge = smoothstep(uThreshold - 0.05, uThreshold + 0.02, f);

  // Normal via finite differences
  float e = 1.5;
  float gx = field(px + vec2(e,0.0)) - field(px - vec2(e,0.0));
  float gy = field(px + vec2(0.0,e)) - field(px - vec2(0.0,e));
  vec3 N = normalize(vec3(-gx, -gy, 0.12));

  // Fresnel
  float fresnel = pow(1.0 - abs(N.z), 2.5);

  // Specular highlights from two lights
  vec3 L1 = normalize(vec3(0.3, 0.6, 1.0));
  vec3 L2 = normalize(vec3(-0.5, -0.3, 0.8));
  float spec1 = pow(max(dot(N, L1), 0.0), 80.0);
  float spec2 = pow(max(dot(N, L2), 0.0), 40.0) * 0.4;

  // Interior depth factor
  float depth = smoothstep(uThreshold, uThreshold + 0.8, f);

  // Subtle caustics inside the glass
  vec2 cUv = px * 0.008 + uTime * 0.3 + N.xy * 15.0;
  float caustic = noise(cUv) * noise(cUv * 1.7 + 5.0);
  caustic = pow(caustic, 1.5) * 0.35;

  // Edge highlight ring
  float ring = smoothstep(uThreshold - 0.02, uThreshold + 0.03, f)
             * (1.0 - smoothstep(uThreshold + 0.03, uThreshold + 0.2, f));

  // Glass color
  vec3 deepBlue = vec3(0.04, 0.12, 0.28);
  vec3 midBlue  = vec3(0.08, 0.22, 0.48);
  vec3 lightBlue = vec3(0.25, 0.55, 0.82);
  vec3 white = vec3(0.85, 0.92, 0.98);

  vec3 col = mix(midBlue, deepBlue, depth * 0.5);
  col += lightBlue * fresnel * 0.6;
  col += white * (spec1 + spec2);
  col += vec3(0.15, 0.45, 0.7) * caustic;
  col += white * ring * 0.5;

  // Alpha: semi-transparent glass
  float alpha = edge * (0.45 + fresnel * 0.3 + depth * 0.1);
  alpha += (spec1 + spec2) * 0.5;
  alpha += ring * 0.25;
  alpha = clamp(alpha, 0.0, 0.92);

  // Premultiplied alpha for correct blending
  gl_FragColor = vec4(col * alpha, alpha);
}`;

export class MetaballRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) throw new Error('WebGL not available');
    this.gl = gl;

    // Compile shaders
    const vs = this._shader(gl.VERTEX_SHADER, VERT);
    const fs = this._shader(gl.FRAGMENT_SHADER, FRAG);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'aPosition');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('Metaball program link error:', gl.getProgramInfoLog(prog));
    }
    this.prog = prog;

    // Uniforms
    this.u = {};
    const uCount = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uCount; i++) {
      const info = gl.getActiveUniform(prog, i);
      const name = info.name.replace('[0]', '');
      this.u[name] = gl.getUniformLocation(prog, info.name);
    }

    // Quad buffer
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, -1,1, 1,1, 1,-1]), gl.STATIC_DRAW);
    const idx = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0,1,2, 0,2,3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    // Config — small spread + high threshold = bubbles fully separate at rest
    this.spread = 10.0;
    this.threshold = 0.88;
    this.blobs = [];
    this.cursor = { x: -9999, y: -9999, radius: 55, active: false };
    this.time = 0;
  }

  _shader(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader error:', gl.getShaderInfoLog(s));
    }
    return s;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /**
   * @param {Array} blobs - [{cx, cy, hw, hh, radius}] in CSS pixels (viewport coords)
   */
  setBlobs(blobs) {
    this.blobs = blobs;
  }

  /**
   * @param {number} x - cursor X in CSS viewport px
   * @param {number} y - cursor Y in CSS viewport px
   * @param {number} radius - cursor influence radius in CSS px
   */
  setCursor(x, y, radius, active) {
    this.cursor = { x, y, radius: radius || 55, active: !!active };
  }

  render(time) {
    this.time = time;
    this.resize();
    const gl = this.gl;
    const dpr = this.canvas.width / this.canvas.clientWidth;

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.prog);

    gl.uniform2f(this.u.uResolution, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.uTime, time);
    gl.uniform1f(this.u.uSpread, this.spread * dpr);
    gl.uniform1f(this.u.uThreshold, this.threshold);

    // Cursor blob (convert to canvas coords, flip Y)
    const cur = this.cursor;
    gl.uniform3f(
      this.u.uCursor,
      cur.x * dpr,
      (this.canvas.clientHeight - cur.y) * dpr,
      cur.radius * dpr
    );
    gl.uniform1f(this.u.uCursorActive, cur.active ? 1.0 : 0.0);

    const count = Math.min(this.blobs.length, 24);
    gl.uniform1i(this.u.uCount, count);

    const blobData = new Float32Array(24 * 4);
    const radiiData = new Float32Array(24);
    for (let i = 0; i < count; i++) {
      const b = this.blobs[i];
      // Convert from CSS viewport coords to canvas pixel coords
      // Also flip Y since WebGL has Y=0 at bottom
      blobData[i * 4 + 0] = b.cx * dpr;
      blobData[i * 4 + 1] = (this.canvas.clientHeight - b.cy) * dpr;
      blobData[i * 4 + 2] = b.hw * dpr;
      blobData[i * 4 + 3] = b.hh * dpr;
      radiiData[i] = (b.radius || 12) * dpr;
    }
    gl.uniform4fv(this.u.uBlobs, blobData);
    gl.uniform1fv(this.u.uRadii, radiiData);

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }
}
