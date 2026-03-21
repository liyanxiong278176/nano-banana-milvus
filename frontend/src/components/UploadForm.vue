<template>
  <div class="upload-form">
    <!-- Upload Area -->
    <div
      class="upload-area"
      :class="{ 'dragover': isDragOver, 'has-file': previewImage }"
      @drop.prevent="handleDrop"
      @dragover.prevent="isDragOver = true"
      @dragleave.prevent="isDragOver = false"
      @click="selectFile"
    >
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        @change="handleFileSelect"
        style="display: none"
      >

      <div v-if="!previewImage" class="upload-prompt">
        <div class="upload-icon">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <path d="M24 32V16M24 16L16 24M24 16L32 24" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="2"/>
          </svg>
        </div>
        <h3 class="upload-title">拖拽图片到这里</h3>
        <p class="upload-hint">或点击选择文件</p>
      </div>

      <div v-else class="upload-preview">
        <img :src="previewImage" alt="Preview" class="preview-image">
        <button class="remove-btn" @click.stop="removeFile">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- Form Fields -->
    <div class="form-fields">
      <div class="form-row">
        <div class="form-group">
          <label class="form-label">商品品类</label>
          <select v-model="form.category" class="form-select" :disabled="loading">
            <option value="">选择品类</option>
            <option v-for="cat in categories" :key="cat.value" :value="cat.value">{{ cat.label }}</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">商品风格</label>
          <select v-model="form.style" class="form-select" :disabled="loading">
            <option value="">选择风格</option>
            <option v-for="style in styles" :key="style.value" :value="style.value">{{ style.label }}</option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label class="form-label">季节</label>
          <select v-model="form.season" class="form-select" :disabled="loading">
            <option value="all_season">全季</option>
            <option value="spring">春季</option>
            <option value="summer">夏季</option>
            <option value="autumn">秋季</option>
            <option value="winter">冬季</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">场景提示（可选）</label>
          <input
            v-model="form.sceneHint"
            type="text"
            class="form-input"
            placeholder="例如：温馨的咖啡厅场景"
            :disabled="loading"
          >
        </div>
      </div>

      <!-- 高级选项 -->
      <div class="advanced-options">
        <div class="advanced-header" @click="showAdvanced = !showAdvanced">
          <span>高级选项</span>
          <svg :class="{ 'rotated': showAdvanced }" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6L8 10M12 6L8 10M8 10V14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div v-show="showAdvanced" class="advanced-content">
          <div class="form-group">
            <label class="checkbox-label">
              <input
                type="checkbox"
                v-model="form.enableQualityCheck"
                :disabled="loading"
                class="checkbox"
              >
              <span>启用 AI 质量评估</span>
            </label>
            <p class="option-hint">启用后将生成图片并进行 AI 质量评分</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Progress -->
    <div v-if="loading" class="progress-container">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      </div>
      <p class="progress-text">{{ progressMessage }}</p>
    </div>

    <!-- Submit Button -->
    <button
      class="submit-btn"
      :disabled="!canSubmit || loading"
      @click="handleSubmit"
    >
      <span v-if="!loading">生成宣传图</span>
      <span v-else class="loading-text">
        <span class="spinner"></span>
        处理中...
      </span>
    </button>

    <!-- Error -->
    <div v-if="error" class="error-message">
      {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['generated'])

const fileInput = ref(null)
const isDragOver = ref(false)
const previewImage = ref(null)
const selectedFile = ref(null)
const loading = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const error = ref('')

const categories = ref([
  { value: 'midi_dress', label: '中长裙' },
  { value: 'maxi_dress', label: '长裙' },
  { value: 'mini_dress', label: '短裙' },
  { value: 'skirt', label: '半身裙' },
  { value: 'top', label: '上装' },
  { value: 'pants', label: '裤装' }
])
const styles = ref([
  { value: 'casual', label: '休闲' },
  { value: 'formal', label: '正式' },
  { value: 'sporty', label: '运动' },
  { value: 'elegant', label: '优雅' },
  { value: 'vintage', label: '复古' },
  { value: 'modern', label: '现代' }
])

const form = ref({
  category: '',
  style: '',
  season: 'all_season',
  sceneHint: '',
  enableQualityCheck: false
})

const showAdvanced = ref(false)

const canSubmit = computed(() => {
  return selectedFile.value && form.value.category && form.value.style
})

const selectFile = () => {
  fileInput.value?.click()
}

const handleFileSelect = (e) => {
  const file = e.target.files?.[0]
  if (file && file.type.startsWith('image/')) {
    setFile(file)
  }
}

const handleDrop = (e) => {
  isDragOver.value = false
  const file = e.dataTransfer.files?.[0]
  if (file && file.type.startsWith('image/')) {
    setFile(file)
  }
}

const setFile = (file) => {
  selectedFile.value = file
  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.value = e.target.result
  }
  reader.readAsDataURL(file)
  error.value = ''
}

const removeFile = () => {
  selectedFile.value = null
  previewImage.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const handleSubmit = async () => {
  if (!canSubmit.value) return

  loading.value = true
  error.value = ''
  progress.value = 0
  progressMessage.value = '上传图片中...'

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('category', form.value.category)
    formData.append('style', form.value.style)
    formData.append('season', form.value.season)
    formData.append('scene_hint', form.value.sceneHint)
    formData.append('enable_quality_check', form.value.enableQualityCheck ? 'true' : 'false')
    // 两阶段检索参数（Neo4j多跳推理 + Milvus向量精排）
    formData.append('retrieval_mode', 'two_stage')
    formData.append('sales_top_k', '100')
    // 多跳推理参数（默认启用）
    formData.append('enable_multi_hop', 'true')
    formData.append('max_hops', '3')

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error('上传失败')
    }

    const data = await response.json()

    // 轮询任务状态
    await pollTaskStatus(data.task_id, data.product_id)

  } catch (err) {
    error.value = err.message || '处理失败，请重试'
    loading.value = false
  }
}

const pollTaskStatus = async (taskId, productId) => {
  const maxAttempts = 120 // 最多2分钟
  let attempts = 0
  let hasCompleted = false // 防止重复触发完成事件

  const poll = async () => {
    // 如果已经完成，不再处理
    if (hasCompleted) return

    attempts++

    try {
      const response = await fetch(`/api/tasks/${taskId}`)
      const data = await response.json()

      progress.value = Math.round(data.progress * 100)

      if (data.status === 'pending') {
        progressMessage.value = '等待处理...'
      } else if (data.status === 'processing') {
        const messages = form.value.enableQualityCheck
          ? [
              '分析图片中...',
              '检索相似爆款...',
              '分析风格特征...',
              '生成多张图片...',
              'AI 裁判评分中...',
              '选择最佳图片...'
            ]
          : [
              '分析图片中...',
              '检索相似爆款...',
              '分析风格特征...',
              '生成宣传图...'
            ]
        progressMessage.value = messages[Math.min(Math.floor(progress.value / 20), messages.length - 1)]
      } else if (data.status === 'completed' && !hasCompleted) {
        hasCompleted = true // 标记已完成，防止重复
        progressMessage.value = '生成完成！'
        loading.value = false

        // 清空表单
        removeFile()
        form.value = {
          category: '',
          style: '',
          season: 'all_season',
          sceneHint: '',
          enableQualityCheck: false
        }

        // 发送结果
        emit('generated', {
          productId: data.result.product_id,
          category: data.result.category,
          style: data.result.style,
          retrievedCount: data.result.retrieved_count,
          generatedCount: data.result.generated_count,
          stylePrompt: data.result.style_prompt,
          images: data.result.generated_images,
          originalImage: `/api/output/${data.result.product_id}/${data.result.product_id}_original.png`,
          referenceImages: [
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_1.png`,
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_2.png`,
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_3.png`
          ].filter((_, i) => i < data.result.retrieved_count),
          qualityScores: data.result.quality_scores || null,
          allScores: data.result.all_scores || [],
          qualityEnabled: data.result.quality_enabled || false
        })

        return
      } else if (data.status === 'failed') {
        hasCompleted = true
        throw new Error(data.error || '处理失败')
      }

      if (attempts < maxAttempts && !hasCompleted) {
        setTimeout(poll, 1000)
      } else if (!hasCompleted) {
        hasCompleted = true
        throw new Error('处理超时')
      }

    } catch (err) {
      hasCompleted = true
      error.value = err.message || '处理失败，请重试'
      loading.value = false
    }
  }

  poll()
}
</script>

<style scoped>
.upload-form {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2rem;
}

.upload-area {
  border: 2px dashed var(--border);
  border-radius: 16px;
  padding: 3rem 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.upload-area:hover {
  border-color: var(--accent);
  background: rgba(124, 58, 237, 0.05);
}

.upload-area.dragover {
  border-color: var(--accent);
  background: rgba(124, 58, 237, 0.1);
}

.upload-area.has-file {
  border-style: solid;
  padding: 1rem;
}

.upload-prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.upload-icon {
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--primary-light);
  color: var(--accent);
}

.upload-title {
  font-size: 1.25rem;
  font-weight: 600;
}

.upload-hint {
  color: var(--text-muted);
}

.upload-preview {
  position: relative;
}

.preview-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 12px;
  display: block;
  margin: 0 auto;
}

.remove-btn {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--accent);
  border: none;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s;
}

.remove-btn:hover {
  transform: scale(1.1);
}

/* Form */
.form-fields {
  margin-top: 2rem;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-muted);
}

.form-select,
.form-input {
  padding: 0.875rem 1rem;
  background: var(--primary);
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text);
  font-size: 1rem;
  transition: all 0.3s;
}

.form-select:focus,
.form-input:focus {
  outline: none;
  border-color: var(--accent);
}

.form-select:disabled,
.form-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.form-input::placeholder {
  color: var(--text-muted);
}

/* Progress */
.progress-container {
  margin-top: 2rem;
}

.progress-bar {
  height: 6px;
  background: var(--primary);
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--accent-light));
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-text {
  margin-top: 0.75rem;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.875rem;
}

/* Submit Button */
.submit-btn {
  width: 100%;
  margin-top: 2rem;
  padding: 1rem 2rem;
  background: var(--accent);
  border: none;
  border-radius: 12px;
  color: white;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.submit-btn:hover:not(:disabled) {
  background: var(--accent-light);
  transform: translateY(-2px);
  box-shadow: 0 8px 30px var(--accent-glow);
}

.submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-text {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Advanced Options */
.advanced-options {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}

.advanced-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
  color: var(--text-muted);
  font-size: 0.875rem;
  font-weight: 500;
}

.advanced-header svg {
  transition: transform 0.3s;
}

.advanced-header svg.rotated {
  transform: rotate(180deg);
}

.advanced-content {
  margin-top: 1rem;
  padding-top: 1rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  font-size: 0.9rem;
  color: var(--text);
}

.checkbox {
  width: 18px;
  height: 18px;
  accent-color: var(--accent);
  cursor: pointer;
}

.option-hint {
  margin-top: 0.5rem;
  margin-left: 2.25rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  line-height: 1.4;
}

/* Error */
.error-message {
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 10px;
  color: #fca5a5;
  font-size: 0.875rem;
  text-align: center;
}
</style>
