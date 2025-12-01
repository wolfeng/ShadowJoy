
import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { Activity, Sparkles, Flame, Loader2 } from 'lucide-react';

// --- 类型定义 ---
interface Results {
  image: HTMLCanvasElement | HTMLImageElement | ImageBitmap;
  segmentationMask: HTMLCanvasElement | HTMLImageElement | ImageBitmap;
}

interface FaceLandmark {
  x: number;
  y: number;
  z: number;
}

interface FaceMeshResults {
  multiFaceLandmarks: FaceLandmark[][];
  image: HTMLCanvasElement | HTMLImageElement | ImageBitmap;
}

declare global {
  class SelfieSegmentation {
    constructor(config: { locateFile: (file: string) => string });
    setOptions(options: { modelSelection: number; selfieMode: boolean }): void;
    onResults(callback: (results: Results) => void): void;
    send(data: { image: HTMLVideoElement }): Promise<void>;
    close(): Promise<void>;
  }

  class FaceMesh {
    constructor(config: { locateFile: (file: string) => string });
    setOptions(options: { maxNumFaces: number; refineLandmarks: boolean; minDetectionConfidence: number; minTrackingConfidence: number }): void;
    onResults(callback: (results: FaceMeshResults) => void): void;
    send(data: { image: HTMLVideoElement }): Promise<void>;
    close(): Promise<void>;
  }

  interface Window {
    SelfieSegmentation: typeof SelfieSegmentation;
    FaceMesh: typeof FaceMesh;
  }
}

// --- Shader: 边缘检测 ---
const edgeDetectionVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const edgeDetectionFragmentShader = `
  uniform sampler2D maskTexture;
  uniform sampler2D uAudioTexture; // 音频频谱纹理 (128x1)
  uniform vec2 resolution;
  uniform float uSmoothing;   // Blur Radius
  uniform float uStrokeWidth; // Line Width
  uniform vec3 color;      // 动态线条颜色
  uniform int uMode;       // 4: 流光(Flow), 5: 雨滴(Raindrops), 6: 圣光(Holy Light)
  uniform float uOutlinePos; // 轮廓位置 (0.0 - 1.0)
  uniform float uTime;     // 时间 (用于动画)
  uniform vec2 uHeadPos;   // 头部位置 (0-1 UV坐标)
  varying vec2 vUv;

  // --- Simplex Noise Functions ---
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  // Cosine based palette, 4 vec3 params
  vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ) {
      return a + b*cos( 6.28318*(c*t+d) );
  }

  void main() {
    // 黄金螺旋采样 (Golden Spiral Disk Sampling)
    // 用于将粗糙的蒙版模糊成平滑的场
    
    float total = 0.0;
    float maxRadius = uSmoothing; 
    float steps = 32.0; 
    vec2 pixel = vec2(1.0) / resolution;
    vec2 aspect = vec2(1.0, resolution.x / resolution.y);

    for (float i = 0.0; i < 32.0; i++) {
        float theta = 2.399963 * i;
        float r = sqrt(i / steps) * maxRadius;
        vec2 offset = vec2(cos(theta), sin(theta)) * r * pixel * aspect;
        total += texture2D(maskTexture, vUv + offset).r;
    }
    
    float average = total / steps;
    // 等值线提取: 计算距离中心边缘(uOutlinePos)的距离
    float dist = abs(average - uOutlinePos);
    float line = 0.0;
    vec3 finalColor = color;
    
    if (uMode == 4) {
      // 模式 4: 流光 (Flow)
      vec3 flowCol = palette(vUv.y + uTime * 0.5, 
          vec3(0.5, 0.5, 0.5),
          vec3(0.5, 0.5, 0.5),
          vec3(1.0, 1.0, 1.0),
          vec3(0.263,0.416,0.557)
      );
      finalColor = flowCol;
      float glow = 1.0 - smoothstep(0.0, uStrokeWidth * 4.0, dist);
      line = pow(glow, 2.0); 
    } else if (uMode == 5) {
      // 模式 5: 雨滴 (Raindrops)
      if (dist < uStrokeWidth * 2.0) {
          float scale = 50.0; 
          vec2 st = vUv * scale;
          st.y += uTime * 6.0; // 向下
          vec2 ipos = floor(st);
          vec2 fpos = fract(st);
          float rnd = fract(sin(dot(ipos, vec2(12.9898, 78.233))) * 43758.5453);
          float proximity = 1.0 - smoothstep(0.0, uStrokeWidth * 1.2, dist);
          float radius = 0.0;
          if (rnd > 0.3) { 
             float baseSize = 0.3 + (rnd * 0.2); 
             radius = baseSize * proximity; 
          }
          float d = length(fpos - 0.5);
          float circle = 1.0 - smoothstep(radius, radius + 0.15, d);
          line = circle;
          finalColor = color + vec3(circle * 0.5); 
      } else {
          line = 0.0;
      }
    } else if (uMode == 6) {
      // 模式 6: 圣光 (Holy Light) - 饱满佛焰 (Fuller Mandorla)
      
      vec2 center = uHeadPos; 
      vec2 uv = vUv - center;
      uv.x *= resolution.x / resolution.y; 
      
      // 垂直偏移
      uv.y -= 0.05; 

      // 饱满形状算法: 减小顶部收缩系数 (0.5), 使其更像饱满的桃形/背光
      float shapeDistortion = 1.0 + smoothstep(-0.5, 1.0, uv.y) * 0.5;
      vec2 shapedUV = vec2(uv.x * shapeDistortion, uv.y);
      
      float r = length(shapedUV); 
      float a = atan(uv.y, uv.x); 
      
      // 基础颜色 - 渐变 (金 -> 红)
      vec3 colCore = vec3(1.0, 0.9, 0.6); // 核心亮金
      vec3 colMid = vec3(1.0, 0.5, 0.1);  // 中间橙金
      vec3 colEdge = vec3(0.9, 0.2, 0.2); // 边缘红
      
      // 动态分层
      // 层1: 外轮廓 (Flame Rim)
      // 增加火焰边缘波动
      float wave = sin(a * 10.0 - uTime * 2.0) * 0.015 + sin(a * 25.0 + uTime * 4.0) * 0.005;
      float outerRadius = 0.55 + wave;
      float distOuter = abs(r - outerRadius);
      float rim = smoothstep(0.03, 0.0, distOuter);
      
      // 层2: 内部装饰线
      float innerRadius = 0.42;
      float distInner = abs(r - innerRadius);
      float innerRing = smoothstep(0.015, 0.0, distInner);
      
      // 纹理: 敦煌式火焰流云
      float pattern = snoise(vec2(a * 6.0, r * 10.0 - uTime * 1.5));
      float patternMask = smoothstep(0.1, 0.6, pattern);
      
      // 填充颜色混合
      // 从中心到外层插值
      float t = smoothstep(0.0, outerRadius, r);
      vec3 layerColor = mix(colCore, colMid, smoothstep(0.0, 0.6, t));
      layerColor = mix(layerColor, colEdge, smoothstep(0.6, 1.0, t));
      
      vec3 bgEffect = vec3(0.0);
      
      // 背景填充 (带纹理)
      float bodyFill = smoothstep(outerRadius, outerRadius - 0.1, r);
      // 内部亮度
      bgEffect += layerColor * bodyFill * 0.4; 
      // 纹理叠加
      bgEffect += colCore * patternMask * bodyFill * 0.3; 
      
      // 叠加线条
      bgEffect += colEdge * rim * 2.5;       // 亮外框
      bgEffect += colMid * innerRing * 1.8;  // 亮内框
      
      // 粒子火星
      float sparkNoise = snoise(vec2(uv.x * 5.0, uv.y * 5.0 - uTime * 1.5));
      float sparks = smoothstep(0.7, 0.95, sparkNoise) * (1.0 - smoothstep(outerRadius, outerRadius + 0.2, r));
      bgEffect += vec3(1.0, 0.9, 0.5) * sparks;
      
      // 整体光晕
      float glow = smoothstep(outerRadius + 0.2, outerRadius, r) * 0.3;
      bgEffect += colMid * glow;

      // 遮罩逻辑: 人体遮挡背景
      float bodyMask = smoothstep(0.2, 0.7, average); 
      vec3 backColor = bgEffect * (1.0 - bodyMask);
      
      finalColor = backColor;
      line = length(backColor); 
    }

    gl_FragColor = vec4(finalColor, line);
  }
`;

type EffectType = 'flow' | 'particles' | 'mystic';

const BodyOutlineFilter: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<string>("正在载入资源...");
  
  // 默认设置为 particles (雨滴)
  const [activeEffect, setActiveEffect] = useState<EffectType>('particles');

  // --- 参数控制状态 ---
  // 默认使用 particles 模式的参数
  const [params, setParams] = useState({
    bloomStrength: 2.0, 
    bloomRadius: 0.8,   
    bloomThreshold: 0.0, 
    smoothing: 40.0,  
    strokeWidth: 0.15, 
    color: "#00d9ff",   
    mode: 5 
  });
  
  const paramsRef = useRef(params);
  const [audioState, setAudioState] = useState<'init' | 'running' | 'suspended' | 'error' | 'closed'>('init');
  const hasLoadedRef = useRef(false); // 用于追踪是否已完成首次渲染

  // --- 音频相关 Ref ---
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  // --- 头部追踪 Ref ---
  const headPosRef = useRef(new THREE.Vector2(0.5, 0.6));
  const trackerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trackerCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const faceMeshRef = useRef<FaceMesh | null>(null); // MediaPipe Face Mesh

  const threeRefs = useRef<{
    renderer: THREE.WebGLRenderer;
    composer: EffectComposer;
    bloomPass: UnrealBloomPass;
    maskTexture: THREE.CanvasTexture;
    audioTexture: THREE.DataTexture;
    material: THREE.ShaderMaterial;
    maskCanvas: HTMLCanvasElement;
    maskCtx: CanvasRenderingContext2D | null;
  } | null>(null);

  const videoSizeRef = useRef<{ w: number, h: number } | null>(null);

  // --- 切换效果预设 ---
  const switchEffect = (effect: EffectType) => {
    setActiveEffect(effect);
    if (effect === 'flow') {
      setParams(prev => ({
        ...prev,
        bloomStrength: 2.5,
        bloomRadius: 1.2,
        strokeWidth: 0.08,
        smoothing: 40.0,
        mode: 4, // 流光模式
        color: '#ffffff' 
      }));
    } else if (effect === 'particles') {
      setParams(prev => ({
        ...prev,
        bloomStrength: 2.0, 
        bloomRadius: 0.8,
        strokeWidth: 0.15, 
        smoothing: 40.0,
        mode: 5, // 雨滴/粒子模式
        color: '#00d9ff' 
      }));
    } else if (effect === 'mystic') {
      setParams(prev => ({
        ...prev,
        bloomStrength: 0.2, 
        bloomRadius: 1.0,
        strokeWidth: 0.05, 
        smoothing: 20.0, 
        mode: 6, // 圣光模式
        color: '#ff9900' // 金橙色
      }));
    }
  };

  // --- 尝试恢复音频 ---
  const resumeAudio = async () => {
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      try {
        await audioContextRef.current.resume();
        setAudioState('running');
        console.log("Audio Context Resumed");
      } catch (e) {
        console.error("Resume failed", e);
      }
    }
  };

  // --- 全局点击恢复音频上下文 ---
  useEffect(() => {
    const handleUserGesture = () => {
      resumeAudio();
    };
    window.addEventListener('click', handleUserGesture);
    window.addEventListener('touchstart', handleUserGesture);
    return () => {
      window.removeEventListener('click', handleUserGesture);
      window.removeEventListener('touchstart', handleUserGesture);
    };
  }, []);

  // --- 定时检查音频状态 ---
  useEffect(() => {
    const interval = setInterval(() => {
      if (audioContextRef.current) {
        if (audioContextRef.current.state !== audioState) {
          setAudioState(audioContextRef.current.state as any);
        }
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [audioState]);

  useEffect(() => {
    paramsRef.current = params; 
    
    if (!threeRefs.current) return;
    const { bloomPass, material } = threeRefs.current;
    
    bloomPass.threshold = params.bloomThreshold;
    material.uniforms.uSmoothing.value = params.smoothing;
    material.uniforms.color.value.set(params.color);
    material.uniforms.uMode.value = params.mode;
  }, [params]);

  const updateDimensions = () => {
    if (!videoSizeRef.current || !containerRef.current || !videoRef.current || !canvasContainerRef.current) return;

    const screenW = containerRef.current.clientWidth;
    const screenH = containerRef.current.clientHeight;
    const videoW = videoSizeRef.current.w;
    const videoH = videoSizeRef.current.h;

    const screenRatio = screenW / screenH;
    const videoRatio = videoW / videoH;

    let finalW, finalH;

    if (screenRatio > videoRatio) {
      finalW = screenW;
      finalH = screenW / videoRatio;
    } else {
      finalH = screenH;
      finalW = screenH * videoRatio;
    }

    videoRef.current.style.width = `${finalW}px`;
    videoRef.current.style.height = `${finalH}px`;
    
    canvasContainerRef.current.style.width = `${finalW}px`;
    canvasContainerRef.current.style.height = `${finalH}px`;

    if (threeRefs.current) {
      threeRefs.current.renderer.setSize(finalW, finalH);
      threeRefs.current.composer.setSize(finalW, finalH);
      threeRefs.current.material.uniforms.resolution.value.set(finalW, finalH);
    }
  };

  useEffect(() => {
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let audioStream: MediaStream | null = null;
    let selfieSegmentation: SelfieSegmentation | null = null;
    let animationFrameId: number;
    let isProcessing = false;

    // 初始化头部追踪辅助画布 (64x64) - 用于Fallback
    if (!trackerCanvasRef.current) {
      const tc = document.createElement('canvas');
      tc.width = 64;
      tc.height = 64;
      trackerCanvasRef.current = tc;
      trackerCtxRef.current = tc.getContext('2d', { willReadFrequently: true });
    }

    const initThreeJS = (width: number, height: number) => {
      if (!canvasContainerRef.current) return;

      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = width;  
      maskCanvas.height = height;
      const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });

      const maskTexture = new THREE.CanvasTexture(maskCanvas);
      maskTexture.minFilter = THREE.LinearFilter;
      maskTexture.magFilter = THREE.LinearFilter;

      // 初始化音频数据纹理 (128 bin)
      const audioData = new Uint8Array(128).fill(0);
      const audioTexture = new THREE.DataTexture(audioData, 128, 1, THREE.RedFormat, THREE.UnsignedByteType);
      
      const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio); 
      canvasContainerRef.current.innerHTML = '';
      canvasContainerRef.current.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      const geometry = new THREE.PlaneGeometry(2, 2);

      const material = new THREE.ShaderMaterial({
        uniforms: {
          maskTexture: { value: maskTexture },
          uAudioTexture: { value: audioTexture },
          resolution: { value: new THREE.Vector2(width, height) },
          uSmoothing: { value: params.smoothing },
          uStrokeWidth: { value: params.strokeWidth },
          color: { value: new THREE.Color(params.color) },
          uMode: { value: params.mode },
          uOutlinePos: { value: 0.5 },
          uTime: { value: 0.0 },
          uHeadPos: { value: headPosRef.current }
        },
        vertexShader: edgeDetectionVertexShader,
        fragmentShader: edgeDetectionFragmentShader,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      });

      const plane = new THREE.Mesh(geometry, material);
      scene.add(plane);

      const renderScene = new RenderPass(scene, camera);
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(width, height),
        params.bloomStrength,
        params.bloomRadius,
        params.bloomThreshold
      );

      const composer = new EffectComposer(renderer);
      composer.addPass(renderScene);
      composer.addPass(bloomPass);

      threeRefs.current = {
        renderer,
        composer,
        bloomPass,
        maskTexture,
        audioTexture,
        material,
        maskCanvas,
        maskCtx
      };
      
      updateDimensions();
    };

    // 原有的粗糙头部追踪（备用）
    const trackHeadFallback = (segmentationMask: HTMLCanvasElement | HTMLImageElement | ImageBitmap) => {
      const ctx = trackerCtxRef.current;
      const cvs = trackerCanvasRef.current;
      if (!ctx || !cvs) return;

      // 绘制蒙版到小画布进行分析
      ctx.clearRect(0, 0, 64, 64);
      ctx.drawImage(segmentationMask, 0, 0, 64, 64);
      
      const imgData = ctx.getImageData(0, 0, 64, 64);
      const data = imgData.data;
      
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      let topY = -1;

      for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
          const i = (y * 64 + x) * 4;
          if (data[i] > 100 || data[i+3] > 100) {
            if (topY === -1) topY = y;
            break;
          }
        }
        if (topY !== -1) break;
      }

      if (topY === -1) return; 

      const sampleHeight = 15;
      const endY = Math.min(64, topY + sampleHeight);

      for (let y = topY; y < endY; y++) {
        for (let x = 0; x < 64; x++) {
          const i = (y * 64 + x) * 4;
          if (data[i] > 100 || data[i+3] > 100) {
            sumX += x;
            sumY += y;
            count++;
          }
        }
      }

      if (count > 0) {
        const targetX = (sumX / count) / 64.0;
        const targetY = 1.0 - ((sumY / count) / 64.0);
        headPosRef.current.lerp(new THREE.Vector2(targetX, targetY), 0.1);
      }
    };

    const onResults = (results: Results) => {
      // 收到第一帧结果，关闭 Loading
      if (!hasLoadedRef.current) {
        hasLoadedRef.current = true;
        setStatus("");
      }

      const refs = threeRefs.current;
      if (!refs || !refs.maskCtx) return;

      refs.maskCtx.clearRect(0, 0, refs.maskCanvas.width, refs.maskCanvas.height);
      refs.maskCtx.drawImage(results.segmentationMask, 0, 0, refs.maskCanvas.width, refs.maskCanvas.height);
      refs.maskTexture.needsUpdate = true;
      
      // 更新头部位置 Uniform
      if (paramsRef.current.mode === 6) {
        // 如果 Face Mesh 没就绪，使用 Fallback
        if (!faceMeshRef.current) {
             trackHeadFallback(results.segmentationMask);
        }
        refs.material.uniforms.uHeadPos.value.copy(headPosRef.current);
      }
      
      // 更新时间 Uniform 实现动画
      refs.material.uniforms.uTime.value = performance.now() / 1000.0;

      let punch = 0;

      if (analyserRef.current && dataArrayRef.current && audioContextRef.current?.state === 'running') {
        analyserRef.current.getByteFrequencyData(dataArrayRef.current);
        refs.audioTexture.image.data.set(dataArrayRef.current);
        refs.audioTexture.needsUpdate = true;
        
        let bassSum = 0;
        const bassBinCount = 16; 
        for(let i = 0; i < bassBinCount; i++) {
          bassSum += dataArrayRef.current[i];
        }
        const bassAverage = bassSum / bassBinCount;
        const normalizedBass = Math.min(1.0, (bassAverage / 255.0) * 1.2);
        punch = Math.pow(normalizedBass, 1.8); 
      }

      const baseParams = paramsRef.current;
      const finalOutlinePos = 0.6 - (punch * 0.35);
      refs.material.uniforms.uOutlinePos.value = finalOutlinePos;
      
      const widthBoost = 0.05;
      const finalStrokeWidth = baseParams.strokeWidth + (punch * widthBoost);
      refs.material.uniforms.uStrokeWidth.value = finalStrokeWidth;

      const boost = baseParams.mode === 6 ? punch * 1.0 : punch * 0.5;
      refs.bloomPass.strength = baseParams.bloomStrength + boost;
      refs.bloomPass.radius = baseParams.bloomRadius + (punch * 0.5);

      refs.composer.render();
      isProcessing = false;
    };

    const start = async () => {
      try {
        setStatus("初始化摄像头...");

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
           setStatus("错误: 需要 HTTPS 安全上下文");
           return;
        }

        const videoElement = videoRef.current;
        if (!videoElement) return;

        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            audio: false
          });
        } catch (videoErr: any) {
          console.error("Video Permission Denied:", videoErr);
          if (videoErr.name === 'NotAllowedError') {
             setStatus("请允许访问摄像头 (Permission Denied)");
          } else if (videoErr.name === 'NotFoundError') {
             setStatus("未找到摄像头");
          } else {
             setStatus(`摄像头错误: ${videoErr.message}`);
          }
          return;
        }

        try {
           audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
           
           const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
           if (AudioContextClass) {
              const audioCtx = new AudioContextClass();
              audioContextRef.current = audioCtx;
              setAudioState(audioCtx.state as any);

              const analyser = audioCtx.createAnalyser();
              analyser.fftSize = 256; 
              analyser.smoothingTimeConstant = 0.15; 
              analyserRef.current = analyser;
              dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

              const source = audioCtx.createMediaStreamSource(audioStream);
              source.connect(analyser);

              if (audioCtx.state === 'suspended') {
                audioCtx.resume();
              }
           }
        } catch (audioErr) {
           console.warn("Microphone access failed:", audioErr);
           setAudioState('error');
        }

        videoElement.srcObject = stream;
        
        videoElement.onloadedmetadata = () => {
          videoElement.play().then(() => {
            const vW = videoElement.videoWidth;
            const vH = videoElement.videoHeight;
            videoSizeRef.current = { w: vW, h: vH };
            
            initThreeJS(vW, vH);
            updateDimensions();
            
            setStatus("加载 AI 模型...");
            loadModel();
          }).catch(e => {
             console.error("Play error:", e);
             setStatus("视频播放失败 (Permission Denied?)");
          });
        };

      } catch (err) {
        console.error(err);
        setStatus("未知错误");
      }
    };

    const loadModel = () => {
       // 1. Load Selfie Segmentation (Always used for body masking)
       if (window.SelfieSegmentation) {
          selfieSegmentation = new window.SelfieSegmentation({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
          });

          selfieSegmentation.setOptions({
            modelSelection: 1, 
            selfieMode: false,
          });

          selfieSegmentation.onResults(onResults);
          
          // 2. Load FaceMesh (Lazily or immediately used for Mystic Mode)
          if (window.FaceMesh) {
             const faceMesh = new window.FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
             });
             faceMesh.setOptions({
               maxNumFaces: 1,
               refineLandmarks: true,
               minDetectionConfidence: 0.5,
               minTrackingConfidence: 0.5
             });
             faceMesh.onResults((faceResults) => {
                 if (faceResults.multiFaceLandmarks && faceResults.multiFaceLandmarks.length > 0) {
                     // 获取鼻子尖端 (索引 1 或 4)
                     const nose = faceResults.multiFaceLandmarks[0][1];
                     // MediaPipe 坐标: x(0-1 左->右), y(0-1 上->下)
                     // Shader UV: u(0-1 左->右), v(0-1 下->上)
                     // 视频在 CSS 中水平翻转了，但数据流通常没有翻转
                     // 直接映射: u = x, v = 1.0 - y
                     const targetX = nose.x;
                     const targetY = 1.0 - nose.y;
                     
                     // 稍微增加平滑系数，FaceMesh 比较稳
                     headPosRef.current.lerp(new THREE.Vector2(targetX, targetY), 0.2);
                 }
             });
             faceMeshRef.current = faceMesh;
          }
          
          setStatus("启动视觉引擎..."); 
          requestAnimationFrame(processFrame);
        } else {
          setTimeout(loadModel, 200);
        }
    }

    const processFrame = async () => {
      const videoElement = videoRef.current;
      if (videoElement && selfieSegmentation && !isProcessing && videoElement.readyState >= 2) {
        isProcessing = true;
        
        // 并行处理:
        // 1. 发送给 SelfieSegmentation (总是需要，为了生成身体遮罩)
        const p1 = selfieSegmentation.send({ image: videoElement });
        
        // 2. 发送给 FaceMesh (仅在秘术模式下需要，为了精确坐标)
        let p2 = Promise.resolve();
        if (paramsRef.current.mode === 6 && faceMeshRef.current) {
            p2 = faceMeshRef.current.send({ image: videoElement });
        }

        await Promise.all([p1, p2]).catch(e => console.error(e));
        
        isProcessing = false;
      }
      animationFrameId = requestAnimationFrame(processFrame);
    };

    const checkScriptDeps = () => {
       if (window.SelfieSegmentation && window.FaceMesh) {
          start();
       } else {
          setTimeout(checkScriptDeps, 100);
       }
    }
    
    checkScriptDeps();

    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (audioStream) audioStream.getTracks().forEach(t => t.stop());
      if (selfieSegmentation) selfieSegmentation.close();
      if (faceMeshRef.current) faceMeshRef.current.close();
      cancelAnimationFrame(animationFrameId);
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (threeRefs.current) {
        threeRefs.current.renderer.dispose();
      }
    };
  }, []);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-black overflow-hidden flex items-center justify-center">
      {status && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black text-white transition-opacity duration-500">
          {(!status.includes("错误") && !status.includes("Error") && !status.includes("Denied")) ? (
            <Loader2 className="w-12 h-12 mb-4 animate-spin text-cyan-400" />
          ) : (
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
          )}
          <div className={`text-lg font-medium tracking-wide ${status.includes("错误") ? "text-red-400" : "text-cyan-100/80 animate-pulse"}`}>
            {status}
          </div>
        </div>
      )}

      {/* 底部效果切换菜单 */}
      <div className="absolute bottom-10 z-50 flex gap-4 bg-black/40 backdrop-blur-md p-2 rounded-2xl border border-white/10 overflow-x-auto max-w-[95vw]">
        
        {/* Raindrops 放在第一位 */}
        <button 
          onClick={() => switchEffect('particles')}
          className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-all ${activeEffect === 'particles' ? 'bg-white text-black scale-105' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}
        >
          <Sparkles size={20} className={activeEffect === 'particles' ? "fill-current" : ""} />
          <span className="text-xs font-bold whitespace-nowrap">雨滴 (Raindrops)</span>
        </button>

        <button 
          onClick={() => switchEffect('flow')}
          className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-all ${activeEffect === 'flow' ? 'bg-white text-black scale-105' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}
        >
          <Activity size={20} className={activeEffect === 'flow' ? "fill-current" : ""} />
          <span className="text-xs font-bold whitespace-nowrap">流光 (Flow)</span>
        </button>
        
        <button 
          onClick={() => switchEffect('mystic')}
          className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl transition-all ${activeEffect === 'mystic' ? 'bg-white text-black scale-105' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}
        >
          <Flame size={20} className={activeEffect === 'mystic' ? "fill-current" : ""} />
          <span className="text-xs font-bold whitespace-nowrap">圣光 (Holy Light)</span>
        </button>
      </div>

      {/* 视频和 Three.js 画布 */}
      <div className="relative w-full h-full flex justify-center items-center">
         {/* Video 直接显示，不隐藏，防黑屏 */}
        <video 
          ref={videoRef} 
          className="absolute object-cover z-0 scale-x-[-1]" 
          playsInline 
          muted 
        />
        
        {/* Canvas 容器：层级最高，透明背景 */}
        <div 
            ref={canvasContainerRef} 
            className="absolute z-10 pointer-events-none mix-blend-screen scale-x-[-1]" 
            style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default BodyOutlineFilter;
