
-- Data Sources: World Bank / Kaggle

-- --------------------------------------------
-- TABLES
-- --------------------------------------------

-- Reference table: country classifications
CREATE TABLE countries (
    country_code TEXT PRIMARY KEY,
    region TEXT,
    income_group TEXT
);

-- Main indicators table: economic data by country and year
CREATE TABLE gdp_indicators (
    country_code TEXT,
    country_name TEXT,
    year INTEGER,
    gdp REAL,
    gdp_per_capita REAL,
    population REAL,
    inflation_rate REAL,
    unemployment REAL,
    continent_name TEXT,
    export_pct_gdp REAL,
    import_pct_gdp REAL,
    FOREIGN KEY (country_code) REFERENCES countries(country_code)
);

-- --------------------------------------------
-- ANALYTICAL VIEWS
-- --------------------------------------------

-- Main view: joins economic indicators with country classifications
CREATE VIEW v_gdp_main AS
SELECT DISTINCT
    g.country_name,
    g.country_code,
    g.year,
    g.gdp,
    g.gdp_per_capita,
    g.population,
    g.inflation_rate,
    g.unemployment,
    g."continent_name",
    c.region,
    c.income_group
FROM gdp_indicators g
JOIN countries c ON g.country_code = c.country_code
WHERE g.gdp IS NOT NULL
AND g.gdp != ''
AND g.gdp_per_capita IS NOT NULL
AND g.gdp_per_capita != '';

-- Regional view: average GDP per capita and total GDP by region and year
CREATE VIEW v_regional_gdp AS
SELECT DISTINCT
    g.year,
    c.region,
    AVG(g.gdp_per_capita) as avg_gdp_per_capita,
    SUM(g.gdp) as total_gdp
FROM gdp_indicators g
JOIN countries c ON g.country_code = c.country_code
WHERE g.gdp IS NOT NULL
AND g.gdp != ''
GROUP BY g.year, c.region;

-- Income group view: average GDP per capita by World Bank income classification
CREATE VIEW v_income_group_gdp AS
SELECT DISTINCT
    g.year,
    c.income_group,
    AVG(g.gdp_per_capita) as avg_gdp_per_capita,
    COUNT(DISTINCT g.country_code) as country_count
FROM gdp_indicators g
JOIN countries c ON g.country_code = c.country_code
WHERE g.gdp_per_capita IS NOT NULL
AND g.gdp_per_capita != ''
GROUP BY g.year, c.income_group;

-- Time series view: GDP trends for top 10 economies 2000-2022
CREATE VIEW v_top10_timeseries AS
SELECT DISTINCT
    g.country_name,
    g.year,
    g.gdp,
    g.gdp_per_capita,
    c.region
FROM gdp_indicators g
JOIN countries c ON g.country_code = c.country_code
WHERE g.country_name IN (
    'United States', 'China', 'Japan', 'Germany',
    'United Kingdom', 'India', 'France', 'Italy',
    'Canada', 'Korea, Rep.'
)
AND g.gdp IS NOT NULL
AND g.gdp != '';
