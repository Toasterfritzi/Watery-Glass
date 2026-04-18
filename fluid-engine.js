// ============================================================
// fluid-engine.js — WebGL 2D Navier-Stokes Fluid Simulation
// ============================================================

import {
  baseVertexShader, splatShader, advectionShader, divergenceShader,
  pressureShader, gradientSubtractShader, curlShader, vorticityShader,
  clearShader, displayShader
} from './shaders.js';

// --- Simulation configuration ---
const CONFIG = {
  SIM_RESOLUTION: 256,
  DYE_RESOLUTION: 1024,
  PRESSURE_ITERATIONS: 20,
  CURL: 30,
  SPLAT_RADIUS: 0.25,
  SPLAT_FORCE: 6000,
  VELOCITY_DISSIPATION: 0.97,
  DYE_DISSIPATION: 0.98,
  COLOR_PALETTE: [
    [0.05, 0.30, 0.70],
    [0.10, 0.50, 0.90],
    [0.00, 0.65, 0.85],
    [0.20, 0.75, 0.95],
    [0.02, 0.15, 0.50],
    [0.00, 0.40, 0.65],
  ],
};

export class FluidEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', {
      alpha: true,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
    });
    if (!this.gl) throw new Error('WebGL not supported');

    const gl = this.gl;

    // Enable float textures
    const halfFloat = gl.getExtension('OES_texture_half_float');
    const halfFloatLinear = gl.getExtension('OES_texture_half_float_linear');
    this.halfFloatType = halfFloat ? halfFloat.HALF_FLOAT_OES : gl.UNSIGNED_BYTE;
    this.supportsLinearFiltering = !!halfFloatLinear;

    // Build quad geometry
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    // Compile all shader programs
    this.programs = {};
    this._compileProgram('splat', splatShader);
    this._compileProgram('advection', advectionShader);
    this._compileProgram('divergence', divergenceShader);
    this._compileProgram('pressure', pressureShader);
    this._compileProgram('gradientSubtract', gradientSubtractShader);
    this._compileProgram('curl', curlShader);
    this._compileProgram('vorticity', vorticityShader);
    this._compileProgram('clear', clearShader);
    this._compileProgram('display', displayShader);

    // Initialize framebuffers
    this._initFramebuffers();

    // Mouse / touch state
    this.pointers = [{ id: -1, x: 0.5, y: 0.5, dx: 0, dy: 0, down: false, moved: false, color: [0.1, 0.4, 0.8] }];
    this._setupEvents();

    // Time
    this.time = 0;
    this.lastTime = Date.now();

    // Auto-splat timer for ambient motion
    this.autoSplatTimer = 0;
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  _compileProgram(name, fragmentSource) {
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, this._compileShader(gl.VERTEX_SHADER, baseVertexShader));
    gl.attachShader(program, this._compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.bindAttribLocation(program, 0, 'aPosition');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
    }

    // Cache uniform locations
    const uniforms = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i);
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }
    this.programs[name] = { program, uniforms };
  }

  _createFBO(w, h, internalFormat, format, type, filter) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      texture, fbo, width: w, height: h,
      attach(id) { gl.activeTexture(gl.TEXTURE0 + id); gl.bindTexture(gl.TEXTURE_2D, texture); return id; }
    };
  }

  _createDoubleFBO(w, h, internalFormat, format, type, filter) {
    let fbo1 = this._createFBO(w, h, internalFormat, format, type, filter);
    let fbo2 = this._createFBO(w, h, internalFormat, format, type, filter);
    return {
      width: w, height: h,
      get read() { return fbo1; },
      set read(v) { fbo1 = v; },
      get write() { return fbo2; },
      set write(v) { fbo2 = v; },
      swap() { const t = fbo1; fbo1 = fbo2; fbo2 = t; }
    };
  }

  _getResolution(resolution) {
    let aspectRatio = this.canvas.width / this.canvas.height;
    if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);
    return this.canvas.width > this.canvas.height ? { width: max, height: min } : { width: min, height: max };
  }

  _initFramebuffers() {
    const gl = this.gl;
    const simRes = this._getResolution(CONFIG.SIM_RESOLUTION);
    const dyeRes = this._getResolution(CONFIG.DYE_RESOLUTION);
    const texType = this.halfFloatType;
    const filterType = this.supportsLinearFiltering ? gl.LINEAR : gl.NEAREST;

    this.velocity = this._createDoubleFBO(simRes.width, simRes.height, gl.RGBA, gl.RGBA, texType, filterType);
    this.divergenceFBO = this._createFBO(simRes.width, simRes.height, gl.RGBA, gl.RGBA, texType, gl.NEAREST);
    this.curlFBO = this._createFBO(simRes.width, simRes.height, gl.RGBA, gl.RGBA, texType, gl.NEAREST);
    this.pressure = this._createDoubleFBO(simRes.width, simRes.height, gl.RGBA, gl.RGBA, texType, gl.NEAREST);
    this.dye = this._createDoubleFBO(dyeRes.width, dyeRes.height, gl.RGBA, gl.RGBA, texType, filterType);
  }

  _blit(target) {
    const gl = this.gl;
    if (target == null) {
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  _setupEvents() {
    const canvas = this.canvas;

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const pointer = this.pointers[0];
      const newX = (e.clientX - rect.left) / rect.width;
      const newY = 1.0 - (e.clientY - rect.top) / rect.height;
      pointer.dx = (newX - pointer.x) * 10;
      pointer.dy = (newY - pointer.y) * 10;
      pointer.x = newX;
      pointer.y = newY;
      pointer.moved = true;
    });

    canvas.addEventListener('mousedown', () => { this.pointers[0].down = true; });
    canvas.addEventListener('mouseup', () => { this.pointers[0].down = false; });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touches = e.targetTouches;
      for (let i = 0; i < touches.length && i < this.pointers.length; i++) {
        const rect = canvas.getBoundingClientRect();
        const pointer = this.pointers[i];
        const newX = (touches[i].clientX - rect.left) / rect.width;
        const newY = 1.0 - (touches[i].clientY - rect.top) / rect.height;
        pointer.dx = (newX - pointer.x) * 10;
        pointer.dy = (newY - pointer.y) * 10;
        pointer.x = newX;
        pointer.y = newY;
        pointer.moved = true;
      }
    }, { passive: false });

    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.pointers[0].down = true;
      const rect = canvas.getBoundingClientRect();
      const t = e.targetTouches[0];
      this.pointers[0].x = (t.clientX - rect.left) / rect.width;
      this.pointers[0].y = 1.0 - (t.clientY - rect.top) / rect.height;
    }, { passive: false });

    canvas.addEventListener('touchend', () => { this.pointers[0].down = false; });

    // Listen globally for mouse movement so splats work even when hovering above UI
    window.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const pointer = this.pointers[0];
      const newX = (e.clientX - rect.left) / rect.width;
      const newY = 1.0 - (e.clientY - rect.top) / rect.height;
      pointer.dx = (newX - pointer.x) * 8;
      pointer.dy = (newY - pointer.y) * 8;
      pointer.x = newX;
      pointer.y = newY;
      pointer.moved = true;
    });
  }

  _splat(x, y, dx, dy, color) {
    const gl = this.gl;
    const p = this.programs.splat;
    gl.useProgram(p.program);

    // Splat velocity
    gl.uniform1i(p.uniforms.uTarget, this.velocity.read.attach(0));
    gl.uniform1f(p.uniforms.aspectRatio, this.canvas.width / this.canvas.height);
    gl.uniform2f(p.uniforms.point, x, y);
    gl.uniform3f(p.uniforms.color, dx * CONFIG.SPLAT_FORCE, dy * CONFIG.SPLAT_FORCE, 0.0);
    gl.uniform1f(p.uniforms.radius, this._correctRadius(CONFIG.SPLAT_RADIUS / 100.0));
    this._blit(this.velocity.write);
    this.velocity.swap();

    // Splat dye
    gl.uniform1i(p.uniforms.uTarget, this.dye.read.attach(0));
    gl.uniform3f(p.uniforms.color, color[0], color[1], color[2]);
    this._blit(this.dye.write);
    this.dye.swap();
  }

  _correctRadius(radius) {
    const aspectRatio = this.canvas.width / this.canvas.height;
    if (aspectRatio > 1) radius *= aspectRatio;
    return radius;
  }

  _randomColor() {
    const palette = CONFIG.COLOR_PALETTE;
    const c = palette[Math.floor(Math.random() * palette.length)];
    return [c[0] * 0.6 + Math.random() * 0.15, c[1] * 0.6 + Math.random() * 0.15, c[2] * 0.6 + Math.random() * 0.15];
  }

  _autoSplats(dt) {
    this.autoSplatTimer += dt;
    if (this.autoSplatTimer > 0.15) {
      this.autoSplatTimer = 0;
      const x = Math.random();
      const y = Math.random();
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.0004 + Math.random() * 0.0008;
      this._splat(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed, this._randomColor());
    }
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this._initFramebuffers();
    }
  }

  step() {
    const now = Date.now();
    const dt = Math.min((now - this.lastTime) / 1000, 0.016666);
    this.lastTime = now;
    this.time += dt;

    const gl = this.gl;
    this.resize();

    // Auto-splats for ambient fluid motion
    this._autoSplats(dt);

    // Process mouse interactions
    for (const pointer of this.pointers) {
      if (pointer.moved) {
        pointer.moved = false;
        this._splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color);
      }
    }

    // --- Curl ---
    const curlP = this.programs.curl;
    gl.useProgram(curlP.program);
    gl.uniform2f(curlP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(curlP.uniforms.uVelocity, this.velocity.read.attach(0));
    this._blit(this.curlFBO);

    // --- Vorticity ---
    const vortP = this.programs.vorticity;
    gl.useProgram(vortP.program);
    gl.uniform2f(vortP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(vortP.uniforms.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(vortP.uniforms.uCurl, this.curlFBO.attach(1));
    gl.uniform1f(vortP.uniforms.curl, CONFIG.CURL);
    gl.uniform1f(vortP.uniforms.dt, dt);
    this._blit(this.velocity.write);
    this.velocity.swap();

    // --- Divergence ---
    const divP = this.programs.divergence;
    gl.useProgram(divP.program);
    gl.uniform2f(divP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(divP.uniforms.uVelocity, this.velocity.read.attach(0));
    this._blit(this.divergenceFBO);

    // --- Clear pressure ---
    const clrP = this.programs.clear;
    gl.useProgram(clrP.program);
    gl.uniform1i(clrP.uniforms.uTexture, this.pressure.read.attach(0));
    gl.uniform1f(clrP.uniforms.value, 0.8);
    this._blit(this.pressure.write);
    this.pressure.swap();

    // --- Pressure solve (Jacobi iterations) ---
    const prsP = this.programs.pressure;
    gl.useProgram(prsP.program);
    gl.uniform2f(prsP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(prsP.uniforms.uDivergence, this.divergenceFBO.attach(0));
    for (let i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(prsP.uniforms.uPressure, this.pressure.read.attach(1));
      this._blit(this.pressure.write);
      this.pressure.swap();
    }

    // --- Gradient subtraction ---
    const grdP = this.programs.gradientSubtract;
    gl.useProgram(grdP.program);
    gl.uniform2f(grdP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(grdP.uniforms.uPressure, this.pressure.read.attach(0));
    gl.uniform1i(grdP.uniforms.uVelocity, this.velocity.read.attach(1));
    this._blit(this.velocity.write);
    this.velocity.swap();

    // --- Advect velocity ---
    const advP = this.programs.advection;
    gl.useProgram(advP.program);
    gl.uniform2f(advP.uniforms.texelSize, 1.0 / this.velocity.width, 1.0 / this.velocity.height);
    gl.uniform1i(advP.uniforms.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(advP.uniforms.uSource, this.velocity.read.attach(0));
    gl.uniform1f(advP.uniforms.dt, dt);
    gl.uniform1f(advP.uniforms.dissipation, CONFIG.VELOCITY_DISSIPATION);
    this._blit(this.velocity.write);
    this.velocity.swap();

    // --- Advect dye ---
    gl.uniform2f(advP.uniforms.texelSize, 1.0 / this.dye.width, 1.0 / this.dye.height);
    gl.uniform1i(advP.uniforms.uVelocity, this.velocity.read.attach(0));
    gl.uniform1i(advP.uniforms.uSource, this.dye.read.attach(1));
    gl.uniform1f(advP.uniforms.dissipation, CONFIG.DYE_DISSIPATION);
    this._blit(this.dye.write);
    this.dye.swap();

    // --- Render to screen ---
    const dspP = this.programs.display;
    gl.useProgram(dspP.program);
    gl.uniform2f(dspP.uniforms.texelSize, 1.0 / this.canvas.width, 1.0 / this.canvas.height);
    gl.uniform1i(dspP.uniforms.uTexture, this.dye.read.attach(0));
    gl.uniform1i(dspP.uniforms.uVelocity, this.velocity.read.attach(1));
    gl.uniform1f(dspP.uniforms.time, this.time);
    this._blit(null);
  }

  // Inject a random burst of splats (e.g., on click or periodically)
  burstSplats(count = 5) {
    for (let i = 0; i < count; i++) {
      const x = Math.random();
      const y = Math.random();
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.001 + Math.random() * 0.002;
      this._splat(x, y, Math.cos(angle) * speed, Math.sin(angle) * speed, this._randomColor());
    }
  }
}
