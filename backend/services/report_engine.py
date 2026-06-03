import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportEngine:
    def __init__(self):
        pass

    def _fig_to_b64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64

    def generate_evaluation_report(self, client_data: dict, prediction: dict,
                                    score_data: dict, recommendations: dict) -> dict:
        sections = []

        sections.append({
            "type": "header",
            "title": "Informe de Evaluaci\u00f3n Crediticia",
            "subtitle": "Sistema Experto de Predicci\u00f3n Financiera IA",
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "report_id": f"INF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        })

        sections.append({
            "type": "client_info",
            "data": {
                "id_cliente": client_data.get("id_cliente", "N/A"),
                "ingreso_mensual": f"S/{client_data.get('ingreso_mensual', 0):,.2f}",
                "total_gastos": f"S/{client_data.get('total_gastos', 0):,.2f}",
                "total_activos": f"S/{client_data.get('total_activos', 0):,.2f}",
                "total_deudas": f"S/{client_data.get('total_deudas', 0):,.2f}",
                "num_deudas": client_data.get("num_deudas", 0),
                "monto_solicitado": f"S/{client_data.get('monto_solicitado', 0):,.2f}",
            }
        })

        sections.append({
            "type": "kpi_grid",
            "kpis": [
                {"label": "Score Financiero", "value": score_data.get("total_score", 0), "max": 1000, "color": "#10b981"},
                {"label": "Clasificaci\u00f3n", "value": score_data.get("classification", "N/A").upper(), "color": "#3b82f6"},
                {"label": "Predicci\u00f3n IA", "value": prediction.get("label", "N/A"), "color": "#8b5cf6"},
                {"label": "Confianza", "value": f"{prediction.get('confidence', 0):.1f}%", "color": "#f59e0b"},
                {"label": "Riesgo", "value": recommendations.get("risk_level", "N/A"), "color": "#ef4444"},
                {"label": "Decisi\u00f3n Final", "value": recommendations.get("final_decision", "N/A"), "color": "#6366f1"},
            ]
        })

        sections.append({
            "type": "score_breakdown",
            "data": score_data.get("breakdown", {}),
            "total_score": score_data.get("total_score", 0),
            "classification": score_data.get("classification", "N/A"),
        })

        sections.append({
            "type": "model_metrics",
            "metrics": prediction,
            "feature_importance": None,
        })

        sections.append({
            "type": "recommendations",
            "data": recommendations.get("recommendations", []),
            "reasons": recommendations.get("reasons", []),
            "observations": recommendations.get("observations", []),
        })

        # Generate chart
        chart_b64 = self._generate_score_chart(score_data)
        sections.append({
            "type": "chart",
            "image": chart_b64,
            "title": "Distribuci\u00f3n del Scoring Financiero",
        })

        return {
            "sections": sections,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "2.0.0",
                "system": "FinPredict Pro - IA Financiera",
            }
        }

    def _generate_score_chart(self, score_data: dict) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        breakdown = score_data.get("breakdown", {})
        categories = []
        values = []
        max_vals = []
        colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444',
                  '#6366f1', '#ec4899', '#14b8a6', '#f97316']

        labels_map = {
            "ingreso": "Ingresos", "gastos": "Gastos", "deuda": "Deuda",
            "excedente": "Excedente", "capacidad": "Cap. Pago",
            "num_deudas": "Nro Deudas", "mora": "Mora",
            "endeudamiento": "Endeudamiento", "capital": "Cap. Trabajo"
        }

        for i, (key, val) in enumerate(breakdown.items()):
            categories.append(labels_map.get(key, key))
            values.append(val["score"])
            max_vals.append(val["max"])

        x = np.arange(len(categories))
        width = 0.35

        bars = ax.bar(x, values, width, label='Obtenido', color=colors[:len(categories)])
        ax.bar(x + width, max_vals, width, label='M\u00e1ximo', color='#e5e7eb', alpha=0.5)

        ax.set_ylabel('Puntaje')
        ax.set_title('Desglose de Scoring Financiero')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(max_vals) * 1.2)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return self._fig_to_b64(fig)

    def generate_dashboard_report(self, stats: dict) -> dict:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        if "score_distribution" in stats:
            labels = list(stats["score_distribution"].keys())
            sizes = list(stats["score_distribution"].values())
            colors_pie = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#dc2626']
            axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%',
                           colors=colors_pie[:len(labels)], startangle=90)
            axes[0, 0].set_title('Distribuci\u00f3n de Clasificaci\u00f3n')

        if "monthly_trend" in stats:
            trend = stats["monthly_trend"]
            axes[0, 1].plot(list(trend.keys()), list(trend.values()),
                           marker='o', color='#3b82f6', linewidth=2)
            axes[0, 1].set_title('Tendencia de Aprobaciones')
            axes[0, 1].tick_params(axis='x', rotation=45)

        if "risk_by_category" in stats:
            risk = stats["risk_by_category"]
            cats = list(risk.keys())
            vals = list(risk.values())
            axes[1, 0].barh(cats, vals, color=['#10b981', '#f59e0b', '#ef4444'])
            axes[1, 0].set_title('Riesgo por Categor\u00eda')

        if "kpi_summary" in stats:
            kpi = stats["kpi_summary"]
            axes[1, 1].axis('off')
            text = '\n'.join([f"{k}: {v}" for k, v in kpi.items()])
            axes[1, 1].text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
                           fontfamily='monospace', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Resumen KPIs')

        plt.tight_layout()
        chart_b64 = self._fig_to_b64(fig)

        return {
            "charts": [{"image": chart_b64, "title": "Dashboard Financiero"}],
            "metadata": {"generated_at": datetime.now().isoformat()},
        }
