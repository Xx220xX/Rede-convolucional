Args('work_path', home)
Args('file_image', arquivoContendoImagens)
Args('file_label', arquivoContendoRespostas)
Args('header_image', bytes_remanessentes_imagem)
Args('header_label', bytes_remanessentes_classes)
Args('numero_epocas', Numero_epocas)
Args('numero_imagens', Numero_Imagens)
Args('numero_treino', Numero_ImagensTreino)
Args('numero_fitnes', Numero_ImagensAvaliacao)
Args('numero_classes', Numero_Classes)
Args('sep', 32)
local nome_classes_lua
local sep_lua
sep_lua = ' '
if sep ~= nil then
    sep_lua = sep
end
for _, v in pairs(classes) do
    if nome_classes_lua == nil then
        nome_classes_lua = v
    else
        nome_classes_lua = nome_classes_lua .. sep_lua .. v
    end
end
Args('nome_classes', nome_classes_lua)