<template>
  <div class="app">
    <!-- Navigation -->
    <nav class="nav">
      <div class="nav-inner">
        <a href="#" class="brand">
          <div class="brand-mark">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="#dc5a35" stroke-width="1.5"/>
              <path d="M9 10 L14 20 L19 10" stroke="#dc5a35" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M10.5 13 L17.5 13" stroke="#dc5a35" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
          </div>
          <span class="brand-name">FashionAI</span>
        </a>
        <div class="nav-links">
          <a href="#upload" class="nav-link">生成图片</a>
          <a href="#features" class="nav-link">核心功能</a>
          <a href="#architecture" class="nav-link">架构设计</a>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-inner">
        <div class="hero-text">
          <div class="hero-eyebrow">
            <span class="eyebrow-dot"></span>
            AI 智能 · 多 Agent 协同
          </div>
          <h1 class="hero-title">
            <span class="hero-title-line">AI 驱动的</span>
            <span class="hero-title-line hero-accent">商品宣传图生成</span>
          </h1>
          <p class="hero-desc">
            基于 Milvus 混合向量检索与 LangGraph 多 Agent 协同编排，
            融合 Dense 语义 + BM25 关键词 + 标量过滤三维检索体系，
            循环检索优化机制驱动，一次通过率提升 30%。
          </p>
          <div class="hero-cta">
            <a href="#upload" class="btn btn-primary">
              <span>开始生成</span>
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path d="M9 3L9 15M9 15L4 10M9 15L14 10" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </a>
            <a href="#architecture" class="btn btn-ghost">
              <span>架构设计</span>
            </a>
          </div>
        </div>

        <div class="hero-cards">
          <div class="process-card" v-for="(step, i) in processSteps" :key="i" :style="`--i:${i}`">
            <div class="process-number">{{ String(i+1).padStart(2,'0') }}</div>
            <div class="process-content">
              <div class="process-icon">{{ step.icon }}</div>
              <div class="process-title">{{ step.title }}</div>
              <div class="process-desc">{{ step.desc }}</div>
            </div>
          </div>
          <div class="hero-visual-bg"></div>
        </div>
      </div>

      <!-- Floating stats -->
      <div class="hero-stats">
        <div class="stat-item" v-for="stat in stats" :key="stat.label">
          <div class="stat-value">{{ stat.value }}</div>
          <div class="stat-label">{{ stat.label }}</div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section id="features" class="features">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-eyebrow">核心能力</div>
          <h2 class="section-title">三大技术支柱</h2>
          <p class="section-sub">从检索到生成，构建完整的 AI 商品宣传图流水线</p>
        </div>
        <div class="features-grid">
          <div class="feature-card feature-card--lg" v-for="feature in features" :key="feature.title">
            <div class="feature-accent-bar" :style="`background:${feature.color}`"></div>
            <div class="feature-header">
              <div class="feature-icon-wrap" :style="`background:${feature.pale}`">
                <span class="feature-icon">{{ feature.icon }}</span>
              </div>
              <div class="feature-badge" :style="`color:${feature.color};background:${feature.pale}`">
                {{ feature.tag }}
              </div>
            </div>
            <h3 class="feature-title">{{ feature.title }}</h3>
            <p class="feature-desc">{{ feature.desc }}</p>
            <ul class="feature-list">
              <li v-for="item in feature.items" :key="item">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <circle cx="7" cy="7" r="2" :fill="feature.color"/>
                </svg>
                {{ item }}
              </li>
            </ul>
            <div class="feature-metric">
              <span class="metric-num" :style="`color:${feature.color}`">{{ feature.metric }}</span>
              <span class="metric-label">{{ feature.metricLabel }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Upload Section -->
    <section id="upload" class="upload-section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-eyebrow">开始使用</div>
          <h2 class="section-title">生成宣传图</h2>
          <p class="section-sub">上传商品图片，AI 自动分析风格并生成专业宣传素材</p>
        </div>
        <UploadForm @generated="handleGenerated" />
      </div>
    </section>

    <!-- Results Section -->
    <section v-if="results.length > 0" class="results-section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-eyebrow">生成成果</div>
          <h2 class="section-title">为您生成的商品宣传图</h2>
        </div>
        <ResultsGallery :results="results" />
      </div>
    </section>

    <!-- Architecture Section -->
    <section id="architecture" class="arch-section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-eyebrow">架构设计</div>
          <h2 class="section-title">系统架构一览</h2>
          <p class="section-sub">三大核心模块的架构设计图（点击查看大图）</p>
        </div>
        <div class="arch-grid">
          <a
            v-for="arch in architectures"
            :key="arch.title"
            :href="arch.png"
            target="_blank"
            class="arch-card"
          >
            <div class="arch-preview">
              <img :src="arch.png" :alt="arch.title" class="arch-img">
              <div class="arch-overlay">
                <span class="arch-view-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" stroke="white" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                </span>
                <span class="arch-view-text">查看大图</span>
              </div>
            </div>
            <div class="arch-info">
              <h4 class="arch-title">{{ arch.title }}</h4>
              <p class="arch-sub">{{ arch.sub }}</p>
            </div>
          </a>
        </div>
      </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
      <div class="footer-inner">
        <div class="footer-top">
          <div class="footer-brand">
            <svg width="24" height="24" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="#dc5a35" stroke-width="1.5"/>
              <path d="M9 10 L14 20 L19 10" stroke="#dc5a35" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M10.5 13 L17.5 13" stroke="#dc5a35" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            <span>FashionAI</span>
          </div>
          <p class="footer-desc">
            基于 LangGraph 多 Agent 编排、Milvus 混合检索与循环优化机制的
            AI 商品宣传图生成平台
          </p>
        </div>
        <div class="footer-bottom">
          <div class="footer-divider"></div>
          <div class="footer-links">
            <span class="footer-copy">© 2026 FashionAI</span>
            <a href="https://deerflow.tech" target="_blank" class="deerflow-link">
              Created By <strong>Deerflow</strong>
            </a>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import UploadForm from './components/UploadForm.vue'
import ResultsGallery from './components/ResultsGallery.vue'

const results = ref([])

const handleGenerated = (result) => {
  const existingIndex = results.value.findIndex(r => r.productId === result.productId)
  if (existingIndex === -1) {
    results.value.unshift(result)
  }
}

const processSteps = [
  { icon: '↑', title: 'Upload', desc: '上传商品平铺图' },
  { icon: '◎', title: 'Retrieval', desc: 'Milvus混合检索' },
  { icon: '◈', title: 'Generation', desc: 'AI生图引擎' },
  { icon: '◉', title: 'QualityJudge', desc: '质量评估评分' },
  { icon: '✓', title: 'Result', desc: '输出最佳结果' },
]

const stats = [
  { value: '35%', label: '容错性提升' },
  { value: '90%', label: 'Top10召回率' },
  { value: '30%', label: '生图通过率提升' },
  { value: '3维', label: '混合检索体系' },
]

const features = [
  {
    title: 'LangGraph 多 Agent 编排',
    icon: '◈',
    tag: '编排框架',
    color: '#7c3aed',
    pale: '#f3f0ff',
    desc: '基于状态机的 5 个智能体协同编排方案，通过条件边实现质量评估后的自动重试机制。',
    items: [
      'Upload → Retrieval → Gen → QualityJudge → Result 状态流',
      '条件边实现质量不达标时自动重试（最多1次）',
      '容错性较传统顺序执行架构提升 35%',
    ],
    metric: '+35%',
    metricLabel: '容错性提升',
  },
  {
    title: 'Milvus 混合检索架构',
    icon: '◎',
    tag: '向量检索',
    color: '#0d9488',
    pale: '#f0fdfb',
    desc: '融合 Dense 向量（语义匹配）+ BM25 稀疏向量（关键词匹配）+ 标量过滤（品类/销量筛选）。',
    items: [
      'Dense: embedding 模型语义编码',
      'BM25: 关键词精确匹配权重',
      '标量过滤: 品类 + 销量范围',
      '权重融合: DENSE×0.6 + BM25×0.3 + SCALAR×0.1',
    ],
    metric: '90%',
    metricLabel: 'Top10 召回率',
  },
  {
    title: '循环检索优化机制',
    icon: '◉',
    tag: '质量优化',
    color: '#dc5a35',
    pale: '#fff4f0',
    desc: '「检索 → LLM 评分 → 查询改写 → 再检索」自适应循环优化，最多 3 轮，质量评分阈值 7.0。',
    items: [
      '评分 < 7.0 时自动触发查询改写',
      '最多 3 轮自适应循环优化',
      '避免低质量参考图导致生图效果差',
    ],
    metric: '+30%',
    metricLabel: '生图一次通过率提升',
  },
]

const architectures = [
  {
    title: 'LangGraph 多 Agent 编排架构',
    sub: '状态机工作流 · 条件边重试机制',
    png: 'architecture_langgraph_cn.png',
  },
  {
    title: 'Milvus 混合检索架构',
    sub: 'Dense + BM25 + 标量三维融合',
    png: 'architecture_milvus_cn.png',
  },
  {
    title: '循环检索优化机制',
    sub: '自适应查询改写 · 最多3轮',
    png: 'architecture_retrieval_cn.png',
  },
]
</script>

<style scoped>
.app {
  min-height: 100vh;
}

/* ===== NAVIGATION ===== */
.nav {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(250, 248, 245, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-light);
}
.nav-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
  height: 68px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: var(--text-primary);
}
.brand-name {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}
.nav-links {
  display: flex;
  align-items: center;
  gap: 2.5rem;
}
.nav-link {
  text-decoration: none;
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-weight: 500;
  transition: color 0.2s;
  letter-spacing: 0.01em;
}
.nav-link:hover { color: var(--accent-coral); }

/* ===== HERO ===== */
.hero {
  max-width: 1200px;
  margin: 0 auto;
  padding: 5rem 2rem 3rem;
  position: relative;
}
.hero-inner {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5rem;
  align-items: center;
  padding-bottom: 3rem;
}
.hero-eyebrow {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent-coral);
  margin-bottom: 1.5rem;
}
.eyebrow-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-coral);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}
.hero-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: clamp(2.8rem, 5vw, 4rem);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.02em;
  margin-bottom: 1.5rem;
}
.hero-title-line { display: block; }
.hero-accent {
  background: linear-gradient(135deg, var(--accent-coral), #f4a488);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-style: italic;
}
.hero-desc {
  font-size: 1.05rem;
  color: var(--text-secondary);
  line-height: 1.8;
  margin-bottom: 2.5rem;
  max-width: 480px;
}
.hero-cta {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0.85rem 2rem;
  border-radius: 100px;
  font-weight: 600;
  font-size: 0.95rem;
  text-decoration: none;
  transition: all 0.25s ease;
  cursor: pointer;
  border: none;
  letter-spacing: 0.01em;
}
.btn-primary {
  background: var(--accent-coral);
  color: white;
  box-shadow: var(--shadow-warm);
}
.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(220,90,53,0.25);
}
.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
  border: 1.5px solid var(--border);
}
.btn-ghost:hover {
  border-color: var(--accent-coral);
  color: var(--accent-coral);
}

/* Process cards */
.hero-cards {
  position: relative;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.75rem;
}
.process-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-light);
  border-radius: 16px;
  padding: 1.25rem 0.75rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  position: relative;
  z-index: 2;
  transition: all 0.25s ease;
  animation: fadeSlideUp 0.5s ease-out both;
  animation-delay: calc(var(--i) * 0.1s);
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
.process-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-md);
  border-color: var(--accent-coral-light);
}
.process-number {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent-coral);
  opacity: 0.3;
  line-height: 1;
  margin-bottom: 0.75rem;
}
.process-icon {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}
.process-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.25rem;
}
.process-desc {
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.3;
}
.hero-visual-bg {
  position: absolute;
  bottom: -20px;
  left: 50%;
  transform: translateX(-50%);
  width: 80%;
  height: 60px;
  background: radial-gradient(ellipse, rgba(220,90,53,0.08), transparent);
  z-index: 1;
}

/* Hero stats */
.hero-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border-light);
  border: 1px solid var(--border-light);
  border-radius: 16px;
  overflow: hidden;
  margin-top: 1rem;
}
.stat-item {
  background: var(--bg-surface);
  padding: 1.5rem;
  text-align: center;
  transition: all 0.2s;
}
.stat-item:hover { background: var(--accent-coral-pale); }
.stat-value {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent-coral);
  line-height: 1;
  margin-bottom: 0.4rem;
}
.stat-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  letter-spacing: 0.03em;
}

/* ===== SECTIONS ===== */
.features, .upload-section, .results-section, .arch-section {
  padding: 5rem 2rem;
}
.section-inner {
  max-width: 1200px;
  margin: 0 auto;
}
.section-header {
  text-align: center;
  margin-bottom: 3.5rem;
}
.section-eyebrow {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-coral);
  margin-bottom: 0.75rem;
}
.section-title {
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: clamp(2rem, 4vw, 2.8rem);
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  margin-bottom: 0.75rem;
}
.section-sub {
  font-size: 1.05rem;
  color: var(--text-muted);
  max-width: 480px;
  margin: 0 auto;
}

/* Features Grid */
.features { background: var(--bg-subtle); }
.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
}
.feature-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-light);
  border-radius: 20px;
  padding: 2rem;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  transition: all 0.3s ease;
}
.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.feature-card:hover .feature-accent-bar { width: 100%; }
.feature-accent-bar {
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  border-radius: 0 4px 4px 0;
  transition: width 0.3s ease;
}
.feature-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.25rem;
}
.feature-icon-wrap {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.feature-icon { font-size: 1.4rem; }
.feature-badge {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 100px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.feature-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.75rem;
  letter-spacing: -0.01em;
}
.feature-desc {
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.7;
  margin-bottom: 1.25rem;
}
.feature-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}
.feature-list li {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}
.feature-metric {
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding-top: 1rem;
  border-top: 1px solid var(--border-light);
}
.metric-num {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1;
}
.metric-label {
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* Architecture Grid */
.arch-section { background: var(--bg-base); }
.arch-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
}
.arch-card {
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
  background: var(--bg-surface);
}
.arch-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.arch-preview {
  position: relative;
  overflow: hidden;
  aspect-ratio: 4/3;
  background: var(--bg-subtle);
}
.arch-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.4s ease;
}
.arch-card:hover .arch-img { transform: scale(1.05); }
.arch-overlay {
  position: absolute;
  inset: 0;
  background: rgba(28,25,23,0.5);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  opacity: 0;
  transition: opacity 0.3s ease;
}
.arch-card:hover .arch-overlay { opacity: 1; }
.arch-view-icon { font-size: 2rem; }
.arch-view-text { color: white; font-size: 0.85rem; font-weight: 500; }
.arch-info { padding: 1.25rem; }
.arch-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
}
.arch-sub { font-size: 0.8rem; color: var(--text-muted); }

/* Upload Section */
.upload-section {
  background: var(--bg-subtle);
}

/* Results Section */
.results-section { background: var(--bg-base); }

/* Footer */
.footer {
  background: var(--text-primary);
  color: rgba(255,255,255,0.8);
  padding: 4rem 2rem 2rem;
}
.footer-inner { max-width: 1200px; margin: 0 auto; }
.footer-top { margin-bottom: 2.5rem; }
.footer-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.4rem;
  font-weight: 600;
  color: white;
  margin-bottom: 1rem;
}
.footer-desc {
  font-size: 0.9rem;
  color: rgba(255,255,255,0.5);
  max-width: 500px;
  line-height: 1.7;
}
.footer-bottom { }
.footer-divider {
  height: 1px;
  background: rgba(255,255,255,0.1);
  margin-bottom: 1.5rem;
}
.footer-links {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.footer-copy { font-size: 0.85rem; color: rgba(255,255,255,0.4); }
.deerflow-link {
  color: rgba(255,255,255,0.5);
  text-decoration: none;
  font-size: 0.85rem;
  transition: color 0.2s;
}
.deerflow-link:hover { color: var(--accent-coral-light); }
.deerflow-link strong { color: var(--accent-coral-light); }

/* ===== RESPONSIVE ===== */
@media (max-width: 1024px) {
  .hero-inner { grid-template-columns: 1fr; gap: 3rem; }
  .hero-cards { max-width: 500px; margin: 0 auto; }
  .features-grid { grid-template-columns: 1fr; }
  .arch-grid { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
  .hero { padding: 3rem 1.5rem 2rem; }
  .hero-stats { grid-template-columns: repeat(2, 1fr); }
  .nav-links { display: none; }
  .hero-title { font-size: 2.2rem; }
}
</style>
