import os
import json
import base64
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VisualizationItem:
    """可視化アイテムの統一データ構造"""
    name: str
    file_path: str
    description: str = ""
    category: str = "general"  # "latent_space", "performance", "trajectory", etc.
    format: str = "png"  # "png", "jpg", "svg", "html"
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedEvaluationResult:
    """画像対応の改良版評価結果管理"""

    def __init__(self, experiment_id: int, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.metrics = {}
        self.visualizations = {}
        self.artifacts = {}
        self.raw_data = {}

        # 画像管理用ディレクトリ構造
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.thumbnails_dir = os.path.join(output_dir, 'thumbnails')
        self.reports_dir = os.path.join(output_dir, 'reports')

        self._ensure_directories()

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        for dir_path in [self.plots_dir, self.thumbnails_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def add_metric(self, name: str, value: float, description: str = "",
                   category: str = "general"):
        """評価指標を追加"""
        self.metrics[name] = {
            'value': value,
            'description': description,
            'category': category,
            'type': 'metric'
        }

    def add_visualization(self, name: str, fig_or_path: Union[plt.Figure, str],
                          description: str = "", category: str = "general",
                          format: str = "png", create_thumbnail: bool = True):
        """可視化結果を追加（Figureオブジェクトまたはパスを受付）"""

        if isinstance(fig_or_path, plt.Figure):
            # matplotlib Figureの場合
            file_path = self._save_figure(fig_or_path, name, format)
        elif isinstance(fig_or_path, str) and os.path.exists(fig_or_path):
            # 既存ファイルパスの場合
            file_path = fig_or_path
        else:
            raise ValueError(f"Invalid figure or path: {fig_or_path}")

        # サムネイル生成
        thumbnail_path = None
        if create_thumbnail and format in ['png', 'jpg', 'jpeg']:
            thumbnail_path = self._create_thumbnail(file_path, name)

        # VisualizationItemを作成
        viz_item = VisualizationItem(
            name=name,
            file_path=file_path,
            description=description,
            category=category,
            format=format,
            thumbnail_path=thumbnail_path,
            metadata={
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'created_timestamp': os.path.getctime(file_path) if os.path.exists(file_path) else None
            }
        )

        self.visualizations[name] = viz_item

    def _save_figure(self, fig: plt.Figure, name: str, format: str = "png") -> str:
        """matplotlib Figureを保存"""
        filename = f"{name}_exp{self.experiment_id}.{format}"
        file_path = os.path.join(self.plots_dir, filename)

        fig.savefig(file_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)  # メモリリークを防ぐ

        return file_path

    def _create_thumbnail(self, file_path: str, name: str, size: tuple = (200, 200)) -> str:
        """サムネイル画像を生成"""
        try:
            with Image.open(file_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)

                thumbnail_filename = f"thumb_{name}_exp{self.experiment_id}.png"
                thumbnail_path = os.path.join(self.thumbnails_dir, thumbnail_filename)

                img.save(thumbnail_path, "PNG")
                return thumbnail_path

        except Exception as e:
            print(f"サムネイル生成失敗 ({name}): {e}")
            return None

    def create_comprehensive_report(self, template: str = "default") -> str:
        """包括的なHTMLレポートを生成"""
        if template == "default":
            return self._create_default_html_report()
        elif template == "markdown":
            return self._create_markdown_report()
        else:
            raise ValueError(f"Unsupported template: {template}")

    def _create_default_html_report(self) -> str:
        """デフォルトHTMLレポートを生成"""
        html_content = self._generate_html_template()

        report_filename = f"evaluation_report_exp{self.experiment_id}.html"
        report_path = os.path.join(self.reports_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_path

    def _generate_html_template(self) -> str:
        """HTMLテンプレートを生成"""
        # メトリクスをカテゴリ別に整理
        metrics_by_category = {}
        for name, info in self.metrics.items():
            category = info.get('category', 'general')
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append((name, info))

        # 可視化をカテゴリ別に整理
        viz_by_category = {}
        for name, viz_item in self.visualizations.items():
            category = viz_item.category
            if category not in viz_by_category:
                viz_by_category[category] = []
            viz_by_category[category].append(viz_item)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>実験 {self.experiment_id} 評価レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #444; margin-top: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
                .metric-name {{ font-weight: bold; color: #333; }}
                .metric-value {{ font-size: 1.2em; color: #4CAF50; margin: 5px 0; }}
                .metric-description {{ font-size: 0.9em; color: #666; }}
                .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .viz-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
                .viz-thumbnail {{ max-width: 100%; height: auto; border-radius: 3px; cursor: pointer; }}
                .viz-title {{ font-weight: bold; margin: 10px 0 5px 0; }}
                .viz-description {{ font-size: 0.9em; color: #666; }}
                .category {{ margin: 30px 0; }}
                .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.9); }}
                .modal-content {{ margin: auto; display: block; width: 90%; max-width: 1000px; }}
                .close {{ position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>実験 {self.experiment_id} 評価レポート</h1>
                <p>生成日時: {self._get_current_timestamp()}</p>

                <!-- メトリクス セクション -->
                <h2>📊 評価メトリクス</h2>
                {self._generate_metrics_html(metrics_by_category)}

                <!-- 可視化 セクション -->
                <h2>📈 可視化結果</h2>
                {self._generate_visualizations_html(viz_by_category)}

            </div>

            <!-- 画像拡大表示用のモーダル -->
            <div id="imageModal" class="modal">
                <span class="close">&times;</span>
                <img class="modal-content" id="modalImage">
            </div>

            <script>
                // モーダル表示のJavaScript
                document.addEventListener('DOMContentLoaded', function() {{
                    const modal = document.getElementById('imageModal');
                    const modalImg = document.getElementById('modalImage');
                    const closeBtn = document.getElementsByClassName('close')[0];

                    // サムネイルクリック時の処理
                    document.querySelectorAll('.viz-thumbnail').forEach(function(thumb) {{
                        thumb.addEventListener('click', function() {{
                            modal.style.display = 'block';
                            modalImg.src = this.getAttribute('data-fullsize');
                        }});
                    }});

                    // モーダルを閉じる
                    closeBtn.addEventListener('click', function() {{
                        modal.style.display = 'none';
                    }});

                    modal.addEventListener('click', function(e) {{
                        if (e.target === modal) {{
                            modal.style.display = 'none';
                        }}
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html_content

    def _generate_metrics_html(self, metrics_by_category: Dict[str, List]) -> str:
        """メトリクス用HTMLを生成"""
        html_parts = []

        for category, metrics in metrics_by_category.items():
            html_parts.append(f'<div class="category"><h3>{category.title()}</h3>')
            html_parts.append('<div class="metric-grid">')

            for name, info in metrics:
                html_parts.append(f'''
                <div class="metric-card">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{info["value"]:.4f}</div>
                    <div class="metric-description">{info.get("description", "")}</div>
                </div>
                ''')

            html_parts.append('</div></div>')

        return '\n'.join(html_parts)

    def _generate_visualizations_html(self, viz_by_category: Dict[str, List]) -> str:
        """可視化用HTMLを生成"""
        html_parts = []

        for category, viz_items in viz_by_category.items():
            html_parts.append(f'<div class="category"><h3>{category.title()}</h3>')
            html_parts.append('<div class="viz-grid">')

            for viz_item in viz_items:
                # 相対パスに変換
                rel_full_path = os.path.relpath(viz_item.file_path, self.reports_dir)
                rel_thumb_path = (os.path.relpath(viz_item.thumbnail_path, self.reports_dir)
                                  if viz_item.thumbnail_path else rel_full_path)

                html_parts.append(f'''
                <div class="viz-card">
                    <div class="viz-title">{viz_item.name}</div>
                    <img src="{rel_thumb_path}" alt="{viz_item.name}" 
                         class="viz-thumbnail" data-fullsize="{rel_full_path}">
                    <div class="viz-description">{viz_item.description}</div>
                </div>
                ''')

            html_parts.append('</div></div>')

        return '\n'.join(html_parts)

    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプを取得"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_db_ready_dict(self) -> Dict[str, Any]:
        """DB保存用の辞書を生成"""
        db_dict = {}

        # メトリクスを平坦化
        for name, info in self.metrics.items():
            db_dict[f'eval_{name}'] = info['value']

        # 可視化パスを追加（主要なもののみ）
        for name, viz_item in self.visualizations.items():
            db_dict[f'{name}_path'] = viz_item.file_path

        # レポートパス（生成されている場合）
        report_path = os.path.join(self.reports_dir, f"evaluation_report_exp{self.experiment_id}.html")
        if os.path.exists(report_path):
            db_dict['evaluation_report_path'] = report_path

        return db_dict

    def export_summary(self) -> Dict[str, Any]:
        """サマリーをエクスポート"""
        return {
            'experiment_id': self.experiment_id,
            'metrics_count': len(self.metrics),
            'visualizations_count': len(self.visualizations),
            'metrics_summary': {name: info['value'] for name, info in self.metrics.items()},
            'visualizations_summary': [
                {
                    'name': viz.name,
                    'category': viz.category,
                    'path': viz.file_path,
                    'has_thumbnail': viz.thumbnail_path is not None
                }
                for viz in self.visualizations.values()
            ]
        }


