🏗️ Estrutura do Projeto
O desenvolvimento foi dividido em três grandes etapas de análise e manipulação de dados:

🔍 Fase 1: Diagnóstico Geral (Exploração)
Nesta etapa, mergulhei no Dataset original para entender o perfil da frota disponível. Respondi a perguntas fundamentais como:

Qual a idade e quilometragem média das motos?

Qual a amplitude de preços (da mais barata à mais luxuosa)?

Como o mercado está dividido entre vendedores particulares e revendedores?

🛠️ Fase 2: Enriquecimento e Limpeza
Dados brutos raramente estão prontos. Aqui, o trabalho foi mais técnico:

Cruzamento de Dados: Integrei a base principal com o arquivo companies.csv para identificar as fabricantes de cada modelo.

Geração de Ativos: Exportei o bikes_completed.csv, uma base refinada e pronta para análises mais profundas.

Análise de Propriedade: Investigamos se o número de donos anteriores impacta diretamente no preço e na conservação (quilometragem) dos veículos.

📈 Fase 3: Insights Estratégicos e Recomendação de Compra
A fase final focou em entender o comportamento das fabricantes e definir o "Checklist de Ouro" para compra.

Identifiquei quais fabricantes possuem maior valor agregado e menor desvalorização.

Filtro de Decisão: Criei um algoritmo de recomendação baseado nos critérios de rigor do CEO:

Motos com até 3 anos de uso.

Baixa quilometragem (máx 40.000 km).

Único dono e venda por pessoa física.

Preço abaixo do valor de Showroom (Margem de lucro imediata).
