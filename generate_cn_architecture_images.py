"""
生成架构设计PNG图片及测试报告（中文版本）
使用Playwright将HTML渲染为PNG截图
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def render_html_to_png():
    """渲染HTML文件为PNG图片"""

    output_dir = Path("D:/agent_learning/nano-banana-milvus2")

    # 要处理的HTML文件列表
    html_files = [
        "architecture_langgraph_cn.html",
        "architecture_milvus_cn.html",
        "architecture_retrieval_cn.html",
        "test_report_cn.html"
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        for html_file in html_files:
            html_path = output_dir / html_file
            # PNG文件名: 去掉_cn.html后缀
            png_path = output_dir / html_file.replace('_cn.html', '_cn.png')

            print(f"Processing: {html_file}")
            await page.goto(f"file:///{html_path.as_posix()}")

            # 等待Mermaid图表渲染完成
            try:
                await page.wait_for_selector('.mermaid', timeout=15000)
                await asyncio.sleep(3)  # 给Mermaid额外渲染时间
            except Exception:
                print(f"  Warning: No mermaid diagram found in {html_file}")

            # 截取整个页面
            await page.screenshot(path=str(png_path), full_page=True)
            print(f"  Generated: {png_path.name}")

        await browser.close()
    print("\nAll PNG files generated successfully!")

if __name__ == "__main__":
    asyncio.run(render_html_to_png())
