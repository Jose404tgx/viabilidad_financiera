import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimulationEngine:
    def __init__(self):
        pass

    def simulate_loan(self, data: dict) -> dict:
        monto = data.get("monto_solicitado", 10000)
        tem = data.get("tem", 0.03)
        num_cuotas = data.get("num_cuotas", 12)

        tea = (1 + tem) ** 12 - 1
        cuota = monto * (tem * (1 + tem) ** num_cuotas) / ((1 + tem) ** num_cuotas - 1)

        cronograma = []
        saldo = monto
        total_intereses = 0

        for i in range(1, num_cuotas + 1):
            interes = saldo * tem
            amortizacion = cuota - interes
            saldo = saldo - amortizacion
            total_intereses += interes
            cronograma.append({
                "cuota": i,
                "saldo_inicial": round(saldo + amortizacion, 2),
                "amortizacion": round(amortizacion, 2),
                "interes": round(interes, 2),
                "cuota_total": round(cuota, 2),
                "saldo_final": round(max(saldo, 0), 2),
            })

        return {
            "loan_details": {
                "monto_solicitado": round(monto, 2),
                "tem": tem,
                "tea": round(tea * 100, 2),
                "num_cuotas": num_cuotas,
                "cuota_mensual": round(cuota, 2),
                "total_intereses": round(total_intereses, 2),
                "total_a_pagar": round(monto + total_intereses, 2),
                "costo_total_credito": round(total_intereses / monto * 100, 2),
            },
            "cronograma": cronograma[:60],
        }

    def simulate_risk_scenarios(self, data: dict) -> dict:
        ingreso = data.get("ingreso_mensual", 5000)
        gastos = data.get("total_gastos", 3000)
        deudas = data.get("total_deudas", 10000)

        scenarios = {
            "optimista": {
                "ingreso_estimado": round(ingreso * 1.15, 2),
                "gastos_estimados": round(gastos * 0.95, 2),
                "capacidad_pago_estimada": round((ingreso * 1.15 - gastos * 0.95) / (deudas * 0.05 + 1), 2),
                "escenario": "Crecimiento de ingresos del 15% y reducci\u00f3n de gastos del 5%",
            },
            "realista": {
                "ingreso_estimado": round(ingreso * 1.05, 2),
                "gastos_estimados": round(gastos * 1.03, 2),
                "capacidad_pago_estimada": round((ingreso * 1.05 - gastos * 1.03) / (deudas * 0.05 + 1), 2),
                "escenario": "Crecimiento de ingresos del 5% e incremento de gastos del 3%",
            },
            "pesimista": {
                "ingreso_estimado": round(ingreso * 0.85, 2),
                "gastos_estimados": round(gastos * 1.15, 2),
                "capacidad_pago_estimada": round((ingreso * 0.85 - gastos * 1.15) / (deudas * 0.05 + 1), 2),
                "escenario": "Reducci\u00f3n de ingresos del 15% e incremento de gastos del 15%",
            },
        }

        return scenarios

    def simulate_mora(self, data: dict) -> dict:
        monto = data.get("monto_solicitado", 10000)
        tem = data.get("tem", 0.03)
        cuota = data.get("cuota_estimada", 500)
        mora_inicial = data.get("mora_diaria", 0)

        results = []
        for mes in range(1, 13):
            mora_actual = mora_inicial + mes * 30 if mes > 0 else mora_inicial
            interes_mora = cuota * (tem / 30) * mora_actual
            penalidad = cuota * 0.05 if mora_actual > 30 else 0
            total_mora = interes_mora + penalidad

            results.append({
                "mes": mes,
                "dias_mora": mora_actual,
                "interes_mora": round(interes_mora, 2),
                "penalidad": round(penalidad, 2),
                "total_mora": round(total_mora, 2),
                "deuda_total": round(cuota + total_mora, 2),
            })

        return {
            "simulation": results,
            "summary": {
                "mora_promedio": round(np.mean([r["dias_mora"] for r in results]), 1),
                "costo_total_mora": round(sum(r["total_mora"] for r in results), 2),
                "deuda_total_acumulada": round(sum(r["deuda_total"] for r in results), 2),
            },
        }

    def simulate_amortization(self, data: dict) -> dict:
        monto = data.get("monto_solicitado", 10000)
        tem = data.get("tem", 0.03)
        num_cuotas = data.get("num_cuotas", 12)

        results = []
        for plazo in [6, 12, 18, 24, 36, 48]:
            cuota = monto * (tem * (1 + tem) ** plazo) / ((1 + tem) ** plazo - 1)
            total = cuota * plazo
            interes_total = total - monto
            results.append({
                "plazo": plazo,
                "cuota_mensual": round(cuota, 2),
                "total_pagar": round(total, 2),
                "intereses": round(interes_total, 2),
            })

        return {
            "monto_referencia": round(monto, 2),
            "tem": tem,
            "opciones": results,
            "recomendacion": min(results, key=lambda x: x["cuota_mensual"] + x["intereses"] * 0.3),
        }
