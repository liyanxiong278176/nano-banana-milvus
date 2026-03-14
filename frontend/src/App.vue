<template>
  <div class="app">
    <!-- Navigation -->
    <nav class="nav">
      <div class="nav-container">
        <div class="logo">
          <span class="logo-icon">✦</span>
          <span class="logo-text">FashionAI</span>
        </div>
        <div class="nav-links">
          <a href="#upload" class="nav-link">生成图片</a>
          <a href="#features" class="nav-link">功能</a>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-content">
        <h1 class="hero-title">
          <span class="hero-title-line">AI 驱动的</span>
          <span class="hero-title-line hero-title-accent">商品宣传图生成</span>
        </h1>
        <p class="hero-subtitle">
          基于向量检索和多模态 AI，一键生成专业级商品宣传图。
          上传平铺图，自动匹配爆款风格，秒级生成营销素材。
        </p>
        <div class="hero-actions">
          <a href="#upload" class="btn btn-primary">
            <span>开始生成</span>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M10 4L10 16M10 16L4 10M10 16L16 10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </a>
          <a href="#features" class="btn btn-secondary">了解更多</a>
        </div>
      </div>

      <div class="hero-visual">
        <div class="floating-card card-1">
          <div class="card-image"></div>
          <div class="card-label">上传平铺图</div>
        </div>
        <div class="floating-card card-2">
          <div class="card-image"></div>
          <div class="card-label">AI 风格分析</div>
        </div>
        <div class="floating-card card-3">
          <div class="card-image"></div>
          <div class="card-label">生成宣传图</div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section id="features" class="features">
      <div class="section-header">
        <h2 class="section-title">核心功能</h2>
        <p class="section-subtitle">强大的 AI 技术，简单的操作流程</p>
      </div>
      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">⚡</div>
          <h3 class="feature-title">向量检索</h3>
          <p class="feature-desc">混合向量检索技术，精准匹配历史爆款风格</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🎨</div>
          <h3 class="feature-title">风格分析</h3>
          <p class="feature-desc">多模态 LLM 自动分析场景、灯光、姿势等视觉元素</p>
        </div>
        <div class="feature-card">
          <div class="feature-icon">✨</div>
          <h3 class="feature-title">AI 生图</h3>
          <p class="feature-desc">基于分析结果，生成符合品牌调性的专业宣传图</p>
        </div>
      </div>
    </section>

    <!-- Upload Section -->
    <section id="upload" class="upload-section">
      <div class="section-header">
        <h2 class="section-title">生成宣传图</h2>
        <p class="section-subtitle">上传商品图片，AI 自动生成专业宣传图</p>
      </div>

      <UploadForm @generated="handleGenerated" />
    </section>

    <!-- Results Section -->
    <section v-if="results.length > 0" class="results-section">
      <div class="section-header">
        <h2 class="section-title">生成结果</h2>
        <p class="section-subtitle">为您生成的商品宣传图</p>
      </div>
      <ResultsGallery :results="results" />
    </section>

    <!-- Footer -->
    <footer class="footer">
      <div class="footer-content">
        <div class="footer-brand">
          <span class="logo-icon">✦</span>
          <span class="logo-text">FashionAI</span>
        </div>
        <div class="footer-divider"></div>
        <a href="https://deerflow.tech" target="_blank" class="deerflow-signature">
          <span>Created By </span>
          <span>Deerflow</span>
        </a>
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
  results.value.unshift(result)
}
</script>

<style scoped>
.app {
  min-height: 100vh;
}

/* Navigation */
.nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  background: var(--glass);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  font-size: 1.25rem;
}

.logo-icon {
  color: var(--accent);
}

.logo-text {
  font-family: 'Playfair Display', serif;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  color: var(--text-muted);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.3s;
}

.nav-link:hover {
  color: var(--accent-light);
}

/* Hero */
.hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  padding: 8rem 2rem 4rem;
  max-width: 1200px;
  margin: 0 auto;
  gap: 4rem;
}

.hero-content {
  flex: 1;
  animation: fadeUp 0.8s ease-out;
}

.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.5rem, 6vw, 4rem);
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 1.5rem;
}

.hero-title-line {
  display: block;
}

.hero-title-accent {
  background: linear-gradient(135deg, var(--accent-light), #c4b5fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.125rem;
  color: var(--text-muted);
  line-height: 1.7;
  max-width: 500px;
  margin-bottom: 2rem;
}

.hero-actions {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.875rem 2rem;
  border-radius: 100px;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.3s ease;
  cursor: pointer;
  border: none;
  font-size: 1rem;
}

.btn-primary {
  background: var(--accent);
  color: white;
  box-shadow: 0 4px 20px var(--accent-glow);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px var(--accent-glow);
}

.btn-secondary {
  background: transparent;
  color: var(--text);
  border: 1px solid var(--border);
}

.btn-secondary:hover {
  background: var(--card-bg);
  border-color: var(--accent);
}

/* Hero Visual */
.hero-visual {
  flex: 1;
  position: relative;
  height: 400px;
}

.floating-card {
  position: absolute;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
  backdrop-filter: blur(10px);
  animation: float 6s ease-in-out infinite;
}

.floating-card .card-image {
  width: 160px;
  height: 200px;
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(16, 185, 129, 0.1));
  border-radius: 12px;
  margin-bottom: 0.75rem;
}

.floating-card .card-label {
  font-size: 0.875rem;
  color: var(--text-muted);
  text-align: center;
}

.card-1 {
  top: 20%;
  left: 10%;
  animation-delay: 0s;
}

.card-2 {
  top: 40%;
  right: 20%;
  animation-delay: 2s;
}

.card-3 {
  bottom: 10%;
  left: 30%;
  animation-delay: 4s;
}

/* Features */
.features {
  padding: 6rem 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.section-header {
  text-align: center;
  margin-bottom: 4rem;
}

.section-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.section-subtitle {
  color: var(--text-muted);
  font-size: 1.125rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2rem;
}

.feature-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem;
  text-align: center;
  transition: all 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-4px);
  border-color: var(--accent);
  box-shadow: 0 10px 40px var(--accent-glow);
}

.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.feature-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.feature-desc {
  color: var(--text-muted);
  line-height: 1.6;
}

/* Upload Section */
.upload-section {
  padding: 6rem 2rem;
  max-width: 900px;
  margin: 0 auto;
}

/* Results Section */
.results-section {
  padding: 4rem 2rem 6rem;
  max-width: 1200px;
  margin: 0 auto;
}

/* Footer */
.footer {
  border-top: 1px solid var(--border);
  padding: 2rem;
  background: var(--glass);
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 2rem;
  flex-wrap: wrap;
}

.footer-divider {
  flex: 1;
  height: 1px;
  background: var(--border);
}

.deerflow-signature {
  color: var(--text-muted);
  text-decoration: none;
  font-size: 0.875rem;
  transition: color 0.3s;
}

.deerflow-signature span:last-child {
  color: var(--accent);
  font-weight: 500;
}

.deerflow-signature:hover {
  color: var(--accent);
}

/* Animations */
@keyframes fadeUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-20px);
  }
}

@media (max-width: 768px) {
  .hero {
    flex-direction: column;
    text-align: center;
    padding-top: 10rem;
  }

  .hero-visual {
    display: none;
  }

  .hero-actions {
    justify-content: center;
  }

  .nav-links {
    display: none;
  }
}
</style>
