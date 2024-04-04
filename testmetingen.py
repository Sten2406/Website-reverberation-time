# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:56:04 2024

@author: Sten
"""
#imports 

import matplotlib
matplotlib.use('Agg') #mogelijkheid om te versturen
from flask import Flask, render_template, request #versturen html
import math 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd  # voor opnemen geluid
import seaborn as sns #maken van grafiek(heatmap)
import pandas as pd #maken van tabellen
import os
#begin waarde defineren
app = Flask(__name__)#?
nummer = 0
df= pd.DataFrame({})
df1 = pd.DataFrame({})
coordinates = []
audio_data = np.array([])
plot_path= ''
t=0
z = 0 
t_list = []
lengte = 0 #lengte van de kamer
standev=[]
script_dir = os.path.dirname(os.path.abspath(__file__))
for i in range(3):
    if i ==0:
        directory_name= 'Dataframe'
        
    if i==1:
        directory_name= 'opgeslagen_data'
    if i==2:
        directory_name= 'static'

    path= os.path.join(script_dir, directory_name)
    if not os.path.exists(path):
            os.makedirs(path)
            print(i, 'Gemaakt')
#voor degene die deze code gaat lezen, alle return code is het sturen naar html en alle request.from code is het opvragen van gegevens in html
#elke aparte html.N heeft eigen code verder staan aantekeningen waar dingen niet duidelijk zijn.
#elke @app.route geeft de knoppen weer die op de website staan en wat er gebeurt bij een klik op de website.

#functie voor het bereken van de nagalmtijd
def metingen_t(frequentie):
    #meet constantes
    SAMPLE_RATE = 96000  #Hz
    DURATION = 4  #Seconde
    REFERENCE_PRESSURE = 20e-6   #Wm^-2
    T = 1/frequentie #Seconde
    normale_drop= 1 #db
    z=0
    
    #ontvang alle data voor DURATION seconde
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
    sd.wait()  #wacht tot alles opgenomen is
    
    
    #functie om van intensiteit naar db te gaan
    def db_formule(intensity):
        db = 20*np.log10(intensity/REFERENCE_PRESSURE)
        return db
    
    #functie om lijst met ellementen op te splitsen in kleinere delen
    def separate_array(array, chunk_size):
        return [array[i:i + chunk_size] for i in range(0, len(array), chunk_size)]
    
    #Haal de eerste 0.2 seconde uit de data omdat deze niet goed zijn
    audio_data = audio_data[int(0.2 * SAMPLE_RATE):]
    
    #maakt alle waardes van de data positief en maakt het 1d
    intensity = abs(audio_data).flatten()
    
    #Berekend de tijd lijst
    time = np.arange(0, len(audio_data)) / SAMPLE_RATE
    
    #splist de data op in stukken met de lengte van de trillingstijd
    chunk_size = int(T*SAMPLE_RATE)
    result = separate_array(intensity, chunk_size)

    
    #gaat alle stukken van de data langs
    for i in range(1, len(result)):
        db1 = db_formule(np.max(result[i-1]))
        db2 =db_formule(np.max(result[i]))
        #de eerste if functie checkt voor elk stukje data of de maximale waarde lager is dan de vorige maximale waarde min de standraad drop
        if db2 < db1-normale_drop and z==0:
            tijd_Begin= (i - 1) * T +(np.argmax(result[i - 1]) * T) / (len(result[0])+1) #functie voor het bepalen van de tijd
            begin_db= db1
            
            z=1
        #Als de eerste loop voltooid is check of het nieuwe maxima 10db lager is dan de vorige waarde
        if z==1:
            if db2 <begin_db-10:
                tijd_eind = (i)*T+(np.argmax(result[i])*T)/(len(result[0])+1)#functie voor het bepalen van de tijd
                t=(tijd_eind-tijd_Begin)*6 #*6 omdat de nagalm tijd voor 60 db en niet voor 10 db
                break
            else:
                t= np.nan
      
    #voegt voor elke stukje data de decibel waarde toe aan een lijst
    db=[]
    for i in range(len(intensity)):
        db.append(db_formule(intensity[i]))
    #plot de waarde van db tegen tijd
    plt.clf()
    plt.plot(time, db)
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('dB')
    plt.title('dB Level over Time')
    plot_path ='static/plot.png'
    plt.savefig(plot_path)
    return  plot_path, t, intensity

#zodra de website wordt geopend vraagt het voor lengte en breedte ( zie index.html)
@app.route('/')
def index():
    return render_template('index.html')

#zodra er op verzend knop gedrukt wordt berekend dit alle meet mogelijkheden en stuurt deze naar result.html
@app.route('/berekeningen', methods=['POST'])
def process():
    global lengte, breedte, metingen_per_hokje, frequentie
    aantal_metingen= []
    x=0
    lengte_hokjegoed=[]
    lengte = float(request.form['lengte'])
    frequentie = float(request.form['frequentie'])
    breedte = float(request.form['breedte'])
    metingen_per_hokje = float(request.form['aantal'])  # vraagt voor alle waardes van de website
    N_hokjes = 0
    while True:
       N_hokjes+=1 
       N_Hl = math.sqrt(N_hokjes * lengte / breedte) 
       N_Hb = math.sqrt(N_hokjes * breedte / lengte) 
        
       if N_Hl.is_integer() and N_Hb.is_integer():#zorgt voor dat de hokjes in lengte en breedte altijd een heel getal zijn.
           N_Hl = int(N_Hl)
           N_Hb = int(N_Hb) 
           aantal_metingen.append( int(N_hokjes * metingen_per_hokje)) 
           lengte_hokjegoed.append(math.sqrt(lengte * breedte / N_hokjes))
           x += 1 
       if x == 11:  #als er tien waarde zijn berekend stop dit deel
           break
    return render_template('result.html', aantal_metingen=aantal_metingen, lengte = lengte_hokjegoed)#terug sturen van data naar html

#zodra er op verzend wordt gedrukt in result.html ontvangt dit de meet constante 
@app.route('/hokjes', methods= ["POST"])
def hokjes():
    global N_y, N_x, lengte, breedte, metingen_per_hokje, df
    N_metingen = int(request.form['lengte_metingen'])
    N_hokjesgoed = int(N_metingen/metingen_per_hokje)
    N_x = int(math.sqrt(N_hokjesgoed * lengte / breedte))
    N_y= int(math.sqrt(N_hokjesgoed * breedte / lengte))
    y1 = []
    x1 = []
    lengte = math.sqrt(lengte*breedte/N_hokjesgoed)#is dit nog nodig?
    for i in range(N_y):
        for j in range(N_x): #voegt elke x en y toe aan hun lijst. voor elke x elke y
            y1.append(i)
            x1.append(j)
            
    #maakt een tabel van de gegevens
    df = pd.DataFrame({'lengte (m)': x1, 'breedte (m)': y1, 't': y1})
    
    #maakt een grid met het aantal vakjes, elke vakje is aan en uit te tikken
    grid_html = ""
    for i in range(N_y):
        for j in range(N_x):
            grid_html += f'<div class="box" id="{i}_{j}" onclick="toggleSelection(this)"></div>' #dit is het aan en uit klikken
    #stuurt alle gegevens naar print.hmtl
    return render_template('print.html', aantal_hokjes_lengte=N_x, aantal_hokjes_breedte=N_y, grid_html=grid_html, lengte = lengte)

#als op print.hmtl alle hokjes zijn aangetikt en verstuurd is starten de metingen
@app.route('/plattegrond', methods=['POST'])
def test1():
    global coordinates, df1, N_y
   
    selected_blocks = request.form.get('selected_blocks')#vraagt welke blokjes zijn aangetikt
    
    #de lengte voor een lege selected_block is 2, als de lengte groter is voegt hij alle lose coordinaten toe aan tabel
    if len(selected_blocks) > 2:
        selected_blocks = selected_blocks.strip('[]').split(',')#strip is het weghalen van blokhaken en split, is de split functie bij een komma
        for block in selected_blocks:
            y, x = map(int, block.strip('"').split('_'))#bijv "4_3" print nog een keer als 4 3 met 4=y en 3=x
            coordinates.append((x, N_y - y - 1)) #draait de grid om zodat linksonderin 0,0 is
        df1 = pd.DataFrame(coordinates, columns=['lengte (m)', 'breedte (m)'])
    return render_template('2.html', x=0, y=0) #roept 2.html op

#wanneer er op start metingen wordt gedrukt begint dit 
@app.route('/test2', methods=['POST'])
def test2():
   global nummer, df, df1, z, t_list, metingen_per_hokje,t, frequentie, audio_data, t #dit wordt in een loop opgeroepen per meting wordt het nummer 1 hoger
   
   #check of voor elke x en y een metingen gedaan is 
   if nummer == len(df):
       html_table = df.to_html()
       return render_template('4.html', html_table=html_table)
   #anders haal x en y uit de tabel 
   y= df['lengte (m)'][nummer]
   x= df['breedte (m)'][nummer]
   g=0
   
   #check of het coordinaat ( de x en y) in de tabel staan met aangevinkte vakjes in print.hmtl
   for i in range(len(df1)):
       x2 = df1['breedte (m)'][i]
       y2= df1['lengte (m)'][i]
       #als dit zo is maakt de nagalmtijd nan(Not an number) en gaat naar de volgende x en y
       if x==x2 and y==y2:
           z=0 #dit is de reset van de meting per hokje
           g=1
           df.loc[nummer, 't']= np.nan
           nummer+=1
           if nummer == len(df):
               html_table = df.to_html()
               return render_template('4.html', html_table=html_table)#check of alle metingen gedaan zijn
           y= df['lengte (m)'][nummer]
           x= df['breedte (m)'][nummer]
           return render_template('2.html', x=x, y=y ,t=np.nan)#dit herstart de loop met een andere x en y
           break#weg?
   #als de x en y niet zijn aangevinkt roep de code voor de nagalm tijd op
   if g==0:
           plot_path, t, audio_data=  metingen_t(frequentie)
           return render_template('6.html', x=x, y=y ,t=t, plot=plot_path)

#op 5.html en 6.html is de optie metingen is goed dan wordt dit uitgevoerd dus een correcte meting
@app.route('/herstart')
def herstart():
    global nummer, df, df1, z, t_list, metingen_per_hokje,t, frequentie, standev
    #voeg de berekende nagalmtijd toe aan de lijst en verhoog z met 1
    t_list.append(t)
    z+=1
    
    #als op deze plek alle metingen gedaan zijn reset z en ga naar de volgende plek, voor de gemiddelde nagalm tijd toe aan de tabel
    if z==metingen_per_hokje:
        df.loc[nummer, 't'] = np.mean(t_list)
        t1= np.mean(t_list)
        standdev=[]
        for i in range(metingen_per_hokje):
            standdev.append(abs(t1-t_list[i]))
        standev.append(standdev.mean())
        
        standdev=[]
        t_list=[]
        z=0
        nummer+=1
        
        return render_template('5.html',t=t1, plot=plot_path, nagalmtijd = t)
    #als alle metingen gedaan zijn ga naar 4.html
    if nummer == len(df):
        html_table = df.to_html()
        return render_template('4.html', html_table=html_table)
    y= df['lengte (m)'][nummer]
    x= df['breedte (m)'][nummer]
    return render_template('2.html', x=x, y=y, t=t)

#op 5.html en 6.html is de optie metingen is  niet goed dan wordt dit uitgevoerd, de knop niet goed is een knop op website
@app.route('/nietgoed')
def nietgoed():
    global nummer, df
    #start de metingen opnieuw zonder iets te verandren
    y= df['lengte (m)'][nummer]
    x= df['breedte (m)'][nummer]
    return render_template('2.html', x=x, y=y)

#op 5.html en 6.html is de optie data opslaan dan wordt dit uitgevoerd, dit is ook een knop op de website
@app.route('/opslaan')
def opslaan():
    global plot_path, audio_data, df, z, nummer, t, t_list, lengte
    
    y= df['lengte (m)'][nummer]
    x= df['breedte (m)'][nummer]
    
    #sla de ruwe data en de plot op in een map de naam van de data is de plek en de meting
    np.savetxt(f'opgeslagen_data\data{x}{y}{z}.txt', audio_data)#f string is het toevoegen van variabelen
    plot =  plt.imread('static/plot.png')
    plt.imsave(f'opgeslagen_data\plot{x}{y}{z}.png', plot)
    return render_template('3.html', x=x, y=y ,t=t, plot='static\plot.png')#deit laat het plot weer zien

#zodra alle data gekregen is (4.hmtl) voert dit uit en maakt een heatmap
@app.route('/heatmap', methods=["POST"])
def heatmap():
    global df, standev
    for i in range(len(df)):
        df.loc[i, 'breedte (m)']=df.loc[i, 'breedte (m)']*lengte
        df.loc[i, 'lengte (m)']=df.loc[i, 'lengte (m)']*lengte
    #maak een heatmap van de lengte, breedte en nagalmtijd
    heatmap_data = df.pivot(index='breedte (m)', columns='lengte (m)', values='t')
    #de kleur voor alle gegevens 
    cmap = sns.diverging_palette(220, 20, as_cmap=True)#as_cmap is de legenda
    cmap.set_bad(color='black')#nan is zwart
    plt.clf()
    sns.heatmap(heatmap_data, cmap=cmap)#maakt heatmap
    plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
    plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)#haal weg? of voeg toe dat we in het midden van de grid zitten
    #laat y van 0 tot max lopen
    plt.gca().invert_yaxis()
    plt.title("Heatmap")
    plt.savefig('static/heatmap.png')
    plt.clf() #dit maakt de heatmap leeg voor een volgende meting
    print(standev)
    return render_template('heatmap.html', heatmap_image='static/heatmap.png')

#als er wordt gekozen om de heatmap opteslaan voer dit uit
@app.route('/opslaan_heatmapp')
def opslaan_heatmap():
    #laat de plot uit zijn map
    plot =  plt.imread('static/heatmap.png')
    #sla hem op in de opgslagen map
    plt.imsave('opgeslagen_data\heatmap.png', plot)
    return render_template('8.html')
if __name__ == '__main__': #dit zorgt voor het runnen van de code
    app.run(debug=True) #dit zorgt voor dat de website gelaunched wordt
