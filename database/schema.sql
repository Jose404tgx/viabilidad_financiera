-- ============================================================
-- ESQUEMA DE BASE DE DATOS RELACIONAL
-- Predictor de Viabilidad Financiera
-- Proyecto Universitario - Sustentación Semestral
-- ============================================================
-- Este script muestra la procedencia de los datos utilizados
-- en el dataset CSV. Los datos provienen de una consulta SQL
-- a una base de datos relacional MySQL/PostgreSQL.
-- ============================================================

CREATE DATABASE IF NOT EXISTS financiera_db;
USE financiera_db;

-- Tabla: Clientes
CREATE TABLE clientes (
    id_cliente INT PRIMARY KEY AUTO_INCREMENT,
    nombre VARCHAR(100),
    documento_identidad VARCHAR(20) UNIQUE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla: Ingresos
CREATE TABLE ingresos (
    id_ingreso INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    ingreso_mensual DECIMAL(12,2) NOT NULL,
    mes_referencia DATE NOT NULL,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- Tabla: Gastos
CREATE TABLE gastos (
    id_gasto INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    total_gastos DECIMAL(12,2) NOT NULL,
    total_costos DECIMAL(12,2) NOT NULL,
    mes_referencia DATE NOT NULL,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- Tabla: Activos y Pasivos
CREATE TABLE patrimonio (
    id_patrimonio INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    total_activos DECIMAL(14,2) NOT NULL,
    total_deudas DECIMAL(14,2) NOT NULL,
    capital_trabajo DECIMAL(14,2),
    endeudamiento_patrimonial DECIMAL(5,2),
    fecha_consulta DATE NOT NULL,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- Tabla: Deudas Activas
CREATE TABLE deudas (
    id_deuda INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    entidad VARCHAR(100),
    monto DECIMAL(12,2),
    saldo_pendiente DECIMAL(12,2),
    mora_diaria INT DEFAULT 0,
    fecha_inicio DATE,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- Tabla: Solicitudes de Crédito
CREATE TABLE solicitudes_credito (
    id_solicitud INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    monto_solicitado DECIMAL(12,2) NOT NULL,
    tem DECIMAL(6,4) NOT NULL,
    num_cuotas INT NOT NULL,
    cuota_estimada DECIMAL(10,2),
    monto_propuesto DECIMAL(12,2),
    fecha_solicitud TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- Tabla: Evaluaciones / Resultados
CREATE TABLE evaluaciones (
    id_evaluacion INT PRIMARY KEY AUTO_INCREMENT,
    id_cliente INT NOT NULL,
    viable BOOLEAN NOT NULL,
    score_total INT,
    clasificacion VARCHAR(20),
    probabilidad_ia DECIMAL(6,4),
    modelo_utilizado VARCHAR(50),
    fecha_evaluacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_cliente) REFERENCES clientes(id_cliente)
);

-- ============================================================
-- CONSULTA SQL QUE GENERA EL DATASET
-- ============================================================
-- Esta consulta JOIN todas las tablas para producir
-- el dataset tabular (CSV) usado para entrenar el modelo.
-- ============================================================

SELECT
    c.id_cliente,
    i.ingreso_mensual,
    g.total_gastos,
    g.total_costos,
    p.total_activos,
    p.total_deudas,
    (SELECT COUNT(*) FROM deudas d WHERE d.id_cliente = c.id_cliente) AS num_deudas,
    (i.ingreso_mensual - g.total_gastos - g.total_costos) AS excedente,
    (i.ingreso_mensual - g.total_gastos - g.total_costos) / NULLIF(g.total_gastos + g.total_costos, 0) AS capacidad_pago,
    p.endeudamiento_patrimonial,
    p.capital_trabajo,
    s.monto_solicitado,
    s.tem,
    s.num_cuotas,
    s.cuota_estimada,
    s.monto_propuesto,
    COALESCE((SELECT MAX(d.mora_diaria) FROM deudas d WHERE d.id_cliente = c.id_cliente), 0) AS mora_diaria,
    e.viable
FROM clientes c
LEFT JOIN ingresos i ON c.id_cliente = i.id_cliente
LEFT JOIN gastos g ON c.id_cliente = g.id_cliente
LEFT JOIN patrimonio p ON c.id_cliente = p.id_cliente
LEFT JOIN solicitudes_credito s ON c.id_cliente = s.id_cliente
LEFT JOIN evaluaciones e ON c.id_cliente = e.id_cliente
WHERE i.mes_referencia = (SELECT MAX(mes_referencia) FROM ingresos WHERE id_cliente = c.id_cliente)
ORDER BY c.id_cliente;
