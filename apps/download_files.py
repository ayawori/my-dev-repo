import wget

init = 10260
end = 10266

for i in range(init, end):
    name = 'EB' + str(i) + '.pdf'
    url = 'https://assets.teradata.com/resourceCenter/downloads/Brochures/' + name
    print(url)
    wget.download(url, name)