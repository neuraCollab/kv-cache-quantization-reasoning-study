import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // API Routes
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      service: 'KV Cache Trace-Level Diagnostic Study API',
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/taxonomy', (req, res) => {
    res.json({
      version: 'v1',
      categories: [
        { letter: 'A', name: 'Arithmetic', description: 'Numerical or symbolic computation error' },
        { letter: 'B', name: 'Logical', description: 'Premises sound, invalid inference step' },
        { letter: 'C', name: 'Strategy-switch', description: 'Unmotivated mid-derivation abandonment of approach' },
        { letter: 'D', name: 'Hallucination', description: 'Invention of nonexistent theorem or formula' },
        { letter: 'E', name: 'Premature-termination', description: 'Trace cut off without boxed answer' },
        { letter: 'F', name: 'Repetition/loop', description: 'Reasoning step repeated ≥3 times without progress' },
      ],
    });
  });

  app.get('/api/models', (req, res) => {
    res.json({
      models: [
        { id: 'deepseek-r1-distill-qwen-1.5b', hf_id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', dtype: 'bfloat16' },
        { id: 'qwen3-1.7b', hf_id: 'Qwen/Qwen3-1.7B', dtype: 'bfloat16', thinking_mode: true },
        { id: 'deepseek-r1-distill-qwen-7b', hf_id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', dtype: 'bfloat16' },
      ],
    });
  });

  app.get('/api/quant-methods', (req, res) => {
    res.json({
      quant_methods: [
        { id: 'bf16', engine: 'vllm', kv_cache_dtype: 'auto', bits: 16 },
        { id: 'fp8_e5m2', engine: 'vllm', kv_cache_dtype: 'fp8_e5m2', bits: 8 },
        { id: 'fp8_e4m3', engine: 'vllm', kv_cache_dtype: 'fp8_e4m3', bits: 8 },
        { id: 'hqq_int4', engine: 'hf', kv_cache_dtype: 'int4', bits: 4 },
        { id: 'hqq_int2', engine: 'hf', kv_cache_dtype: 'int2', bits: 2 },
      ],
    });
  });

  // Pipeline simulation trigger
  app.post('/api/pipeline/run', (req, res) => {
    const { light_mode = false } = req.body || {};
    res.json({
      status: 'started',
      light_mode,
      phases: ['GENERATE', 'FIND_FDP', 'JUDGE', 'ANALYZE'],
      message: 'Pipeline run initialized successfully.',
    });
  });

  // Vite integration
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
