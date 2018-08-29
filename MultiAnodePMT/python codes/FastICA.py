import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from sklearn.decomposition import FastICA
import heapq
import os
from decimal import Decimal
home_path = os.getcwd().replace('/MultiAnodePMT/python codes','')
data_region='data_array_region1'



#Loads file data
data_array=np.load(home_path+'/processed_data/cs10288/'+data_region+'.npy')





def create_image(data_per_idx,map_file):
    """
    Private method that maps each channel into a pixel in the
    MA-PMT grid and creates its image.

    Args:
    ----
    data_per_idx (array): one realization (trigger/time unit) per channel
    map_file (str): complete path to the mapping file

    Returns:
    -------
    data_matrix (2D array): image of the MA-PMT
    """
    # get mapping for image pixels
    pixel_map = np.loadtxt(map_file)

    data_matrix = np.zeros([8, 8])

    for i in range(8):
        for j in range(8):
            if pixel_map[i, j] == 0:
                continue
            else:
                channel = int(pixel_map[i, j]-1)
                data_matrix[i, j] = data_per_idx[channel]

    return data_matrix





#Path to the channel reading mapping file
map_path = home_path+'/processed_data/cs10288/mapping.txt'





def generate_images(path):
    """
        Private method creating a vector image in a chosen folder
        Only for create image
        
        Args:
        ----
        map_file (str): complete path to where the images will be generated

        Returns:
        Image
        -------
        
    """    
    #Method to generate the images
    for lista in range(165):
        #Calling the function create_image
        ma_pmt_image = create_image(data_array[lista,:], map_path)

        #Sets image size
        plt.figure(figsize=(15,15))
        plt.subplots(1,1)
        #Create the image
        plt.matshow(ma_pmt_image,cmap='YlGnBu', fignum=1)
        X= str(lista+1)

        rm = plt.cm.ScalarMappable(cmap='YlGnBu',norm=plt.Normalize(vmin=np.amin(ma_pmt_image), vmax=np.amax(ma_pmt_image)))
        rm._A = []
       
        cb=plt.colorbar(rm)
        cb.ax.tick_params(labelsize=15)
        
        
        tick_locator = ticker.MaxNLocator(nbins=8)
        cb.locator = tick_locator
        cb.update_ticks()
        

        #Sets image margins size
        plt.title("# Pixel Map "+ X , fontsize = 25)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Pixels X",fontsize=20)
        plt.ylabel("Pixels Y",fontsize=20)

        plt.savefig(path+'/Imagem '+X+'.png', bbox_inches='tight')
        plt.clf()
     




#place to create the image
image_path = home_path+'/analysis_images/sources_image' 





#generate_images(image_path)



#Serpentine mode vectoring
#transform into a vector an array received from Image Channel
def vector_serpentina():
    """
        Private method that returns the transformation
        of an matrix in array using the serpentine method

        Args:
        ----
        None 

        Returns:
        -------
        lista (1D array): vectorization result
    """
    
    lista=np.array([])
    #Receive array of "create_image" function from file Image Channel
    vetor=ma_pmt_image
    for i in range (8):

        if(i==1 or i==3 or i==5 or i==7):
            #Reverse the order of elements in an array along the given axis.
            vetor=np.flip(ma_pmt_image, 1) 
        else:
            #Does not require even lines to flip
            vetor=ma_pmt_image

        for j in range(8):
            lista=np.append(lista,vetor[i,j])
    return lista
        


#Uses the vector_serpentina method to vectorize all data array entries
#It returns a 165X64 matrix, with the values of the Data_Array.npy files mapped
for l in range(165):
    #ma_pmt_image receives array from imported file after passing through "create image"
    ma_pmt_image = create_image(data_array[l,:], map_path)
    if (l==0):
        #Used to initialize matriz_serpentina to avoid error in vstack
        matriz_serpentina=vector_serpentina()
    else:
        #Stack arrays in sequence vertically (row wise).
        matriz_serpentina=np.vstack((  matriz_serpentina, vector_serpentina()  ))
 

#Vectorization patches mode -> Vectorize an array received from Image Channel
#Argument of the function -> Order that we want to vectorize the matrix
'''
  
+-------+-------+
| . . . | . . . |
|Bloco a|Bloco b|
| . . . | . . . |
+-------+-------+
| . . . | . . . |
|Bloco c|Bloco d|
| . . . | . . . |
+-------+-------+
  
'''
def vector_patches(rest):
 """
     Private method that returns the transformation
     of an matriz in array using the patches method

     Args:
     ----
     rest(string):Order that we want to vectorize the matrix
     Ex: order = 'a b c d'
     separated by whitespace
     Returns:
     -------
     result (1D array): vectorization result
 """
 Vector=ma_pmt_image
 #Divide the matrix into 4 parts
 a, b, c, d = Vector[:4, :4], Vector[:4, 4:], Vector[4:, :4], Vector[4:, 4:]
 #flatten returns an array of a matrix
 a=np.array(a).flatten()
 b=np.array(b).flatten()
 c=np.array(c).flatten()
 d=np.array(d).flatten()
 
 
 #Creates a dictionary and vectorizes the rest argument
 order=rest.split( )
 ordena=['a', 'b', 'c', 'd']
 dados=[a,b,c,d]
 #zip This function returns a list of tuples
 dictionary = dict(zip(ordena, dados))

 one=np.append(dictionary[order[0]],dictionary[order[1]])
 two=np.append(one,dictionary[order[2]])
 result=np.append(two,dictionary[order[3]])
 
 return result
   


#Matrix using the patch function in chosen order
#It returns a 165X64 matrix, with the values of the Data_Array.npy files mapped 
ordem='c d a b'
for l in range(165):
    #ma_pmt_image receives matrix from imported file after passing through "create image"
    ma_pmt_image = create_image(data_array[l,:], map_path)
    if (l==0):
        #Used to initialize matriz_serpentina to avoid error in vstack
        matriz_patches=vector_patches(ordem)
    else:
        #Stack arrays in sequence vertically (row wise).
        matriz_patches=np.vstack((  matriz_patches, vector_patches(ordem)  ))


#Desvectorize serpentine mode
def desvector_serpentina(vetor):
    """
        Private method that returns the transformation
        of an array in matrix using the serpentine method

        Args:
        ----
        Vetor(1D Array):Vector that we will transform into matrix

        Returns:
        -------
        v (2D array): transformation result
    """
    #Using the copy, to avoid the error of change in the serpentine matrix
    teste=vetor.copy()
    #Gives a new shape to an array without changing its data.
    v = teste.reshape((8, 8))
    for i in range (8):

        if(i==1 or i==3 or i==5 or i==7):
            bloco = v[i,:8]
            #Reverse the order of elements in an array along the given axis.
            bloco[:] = np.flip(bloco.copy(),0)
        
    return v


#Desvectorize mode patches
#Argument of the function -> order that the matrix has been vectorized, the SAME defined for the function vector_patches
'''
  
+-------+-------+
| . . . | . . . |
|Bloco a|Bloco b|
| . . . | . . . |
+-------+-------+
| . . . | . . . |
|Bloco c|Bloco d|
| . . . | . . . |
+-------+-------+
  
'''
def desvector_patches(rest,vetor):
    """
        Private method that returns the transformation
        of an array in matrix using the patches method

        Args:
        ----
        rest(string):Order that we want to vectorize the matrix
        Ex: order = 'a b c d'  --> the SAME defined for the function vector_patches
        separated by whitespace
        Vetor(1D Array):Vector that we will transform into matrix
        
        Returns:
        -------
        join (2D array): transformation result
    """
    patch=vetor
    #Divide the vector into 4 parts
    first, second,third,fourth = patch[:16], patch[16:32], patch[32:48], patch[48:64]
    
    #Gives a new shape to an array without changing its data.
    first=first.reshape((4,4))
    second=second.reshape((4,4))
    third=third.reshape((4,4))
    fourth=fourth.reshape((4,4))  
   
    
    #Creates a dictionary and vectorizes the rest argument
    order=rest.split( )
    ordena=[order[0],order[1],order[2],order[3]]
    dados=[first,second,third,fourth]
    dictionary = dict(zip(ordena, dados))
    

    #Stack arrays in sequence horizontally (column wise).
    one=np.hstack([dictionary['a'], dictionary['b']])
    two=np.hstack([dictionary['c'],dictionary['d']])
    join=np.vstack([one,two])
    return join


np.array_equal(desvector_serpentina(matriz_serpentina[164]),desvector_patches(ordem,matriz_patches[164]))


# get mapping for image pixels
pixel_map = np.loadtxt(map_path)
#algorithm='parallel'
algorithm='deflation'


def calc_ICA(matriz_method,number_componentes):
    # Compute ICA:
    """
        Private method creating a vector image in a chosen folder
        Only for Fast ICA
        #note that you need the variable " ordem " to desvector_patches
        Args:
        ----
        matriz_method (str): Method that will return a transformation
        of an matrix in array
        number_componentes(int): Number of components that will be used

        Returns:
        pesq(array 2D)
        -------
        
    """  
    rn = np.random.RandomState(0)
    ica = FastICA(n_components=number_componentes,algorithm=algorithm,whiten=True, fun='logcosh',random_state=rn)
    
    
    if (matriz_method=='Serpentine'):
        #Use desvetor_serpentina 
            S_ = ica.fit_transform(matriz_serpentina) 
            A_ = ica.components_             
    if(matriz_method=='Patches'):
        #Use desvetor_patches in matshow
            S_ = ica.fit_transform(matriz_patches) 
            A_ = ica.components_ 


    #Convert the input to an array.
    myarray = np.asarray(S_)
    pesq = np.asarray(A_)
    return pesq 

def notacao_cientica(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


#Path to folders(Sources) with sources numbers
if (algorithm=='parallel'):

    def path_serpentina(number):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        serpentina_path = home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/serpentine_method/Sources' 
        return serpentina_path

    def path_patches(number):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        patches_path= home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/patches_method/Sources'
        return patches_path
if (algorithm=='deflation'):
    
    def path_serpentina(number):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        serpentina_path = home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/serpentine_method/Sources' 
        return serpentina_path

    def path_patches(number):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        patches_path= home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/patches_method/Sources'
        return patches_path


#Sets image size
def create_image_ICA(method,pesq):
    """
        Private method creating a vector image in a chosen folder
        Only for Fast ICA
        #note that you need the variable " ordem " to desvector_patches
        Args:
        ----
        method(string):Method that will be used to transform the matrix in 8 X 8
   
        pesq(array): Matrix returned by function "calc_ICA"
       
        Returns:
        Image
        -------
        
    """  
    number=pesq.shape[0]
    #Method to generate the images
    for lista in range(number):
        
        
        #Sets image size
        plt.figure(figsize=(15,15))
        plt.subplots(1,1)
    
        #Create the image
        if (method=='Serpentine'):
            #Use desvetor_serpentina in matshow
            plt.matshow(desvector_serpentina(pesq[lista]),cmap='YlGnBu', fignum=1,vmin=np.amin(pesq[lista]), vmax=np.amax(pesq[lista]))
            path=path_serpentina(number)
           
        if(method=='Patches'):
            #Use desvetor_serpentina in matshow
            plt.matshow(desvector_patches(ordem,pesq[lista]),cmap='YlGnBu', fignum=1,vmin=np.amin(pesq[lista]), vmax=np.amax(pesq[lista]))
            path=path_patches(number)
           
            
            
        X= str(lista+1)
        N=str(number)
        sm = plt.cm.ScalarMappable(cmap='YlGnBu',norm=plt.Normalize(vmin=np.amin(pesq[lista]), vmax=np.amax(pesq[lista])))
        sm._A = []
        cb=plt.colorbar(sm,format=ticker.FuncFormatter(notacao_cientica))
        cb.ax.tick_params(labelsize=15)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        
        #Sets image margins size
        
        #plt.xlabel('Smarts')
        #plt.ylabel('Probability')
        plt.title('Source '+X +' of '+str(number)+': FastICA Method '+str(method) , fontsize = 25)
        #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
        # hide tick and tick label of the big axes
        plt.xlabel("Pixels X",fontsize=20)
        plt.ylabel("Pixels Y",fontsize=20)

        #place to create the image
        

 
        plt.savefig(path+'/sources_'+N+'_image_'+X+'.png', bbox_inches='tight')
 
        plt.clf()
     

#Method Serpentine

#For 03 Sources
sources03=calc_ICA('Serpentine',3)
create_image_ICA("Serpentine",sources03) 

#For 04 Sources
sources04=calc_ICA('Serpentine',4)
create_image_ICA("Serpentine",sources04)

#For 05 Souces
sources05=calc_ICA("Serpentine",5)
create_image_ICA("Serpentine",sources05) 

#For 06 Sources
sources06=calc_ICA("Serpentine",6)
create_image_ICA("Serpentine",sources06)

#For 07 Sources
sources07=calc_ICA("Serpentine",7)
create_image_ICA("Serpentine",sources07)

#For 08 Sources
sources08=calc_ICA("Serpentine",8)
create_image_ICA("Serpentine",sources08)

#For 09 Sources
sources09=calc_ICA("Serpentine",9)
create_image_ICA("Serpentine",sources09) 

#For 48 Sources
sources48=calc_ICA("Serpentine",48)
create_image_ICA("Serpentine",sources48) 

#For 64 Sources
sources64=calc_ICA("Serpentine",64)
create_image_ICA("Serpentine",sources64) 



#Method Pacthes

#For 03 Sources
psources03=calc_ICA('Patches',3)
create_image_ICA('Patches',psources03) 

#For 04 Sources
psources04=calc_ICA('Patches',4)
create_image_ICA('Patches',psources04) 

#For 05 Souces
psources05=calc_ICA('Patches',5)
create_image_ICA('Patches',psources05) 

#For 06 Sources
psources06=calc_ICA('Patches',6)
create_image_ICA('Patches',psources06) 

#For 07 Sources
psources07=calc_ICA('Patches',7)
create_image_ICA('Patches',psources07) 

#For 08 Sources
psources08=calc_ICA('Patches',8)
create_image_ICA('Patches',psources08) 

#For 09 Sources
psources09=calc_ICA('Patches',9)
create_image_ICA('Patches',psources09) 

#For 48 Sources
psources48=calc_ICA('Patches',48)
create_image_ICA('Patches',psources48) 

#For 64 Sources
psources64=calc_ICA('Patches',64)
create_image_ICA('Patches',psources64) 

#Path to folders(Analysis) with sources numbers

if (algorithm=='parallel'):
    def path_image_analysis_serpentina(number,cont):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        if cont==1:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/serpentine_method/Analysis/3_Warmer_Pixels' 
        if cont==2:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/serpentine_method/Analysis/50% of the value of the Warmer Pixels' 

        return analysis_image_path
    def path_image_analysis_patches(number,cont):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        if cont==1:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/patches_method/Analysis/3_Warmer_Pixels' 
        if cont==2:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/parallel/fastICA_images/'+N+'_Sources/patches_method/Analysis/50% of the value of the Warmer Pixels' 

        return analysis_image_path
    
if (algorithm=='deflation'):
    def path_image_analysis_serpentina(number,cont):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        if cont==1:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/serpentine_method/Analysis/3_Warmer_Pixels' 
        if cont==2:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/serpentine_method/Analysis/50% of the value of the Warmer Pixels' 

        return analysis_image_path
    def path_image_analysis_patches(number,cont):
        #number=Number of components that were used in the function "calc_ICA"
        N=str(number)
        if cont==1:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/patches_method/Analysis/3_Warmer_Pixels' 
        if cont==2:
            analysis_image_path = home_path+'/analysis_images/'+data_region+'/algorithm/deflation/fastICA_images/'+N+'_Sources/patches_method/Analysis/50% of the value of the Warmer Pixels' 

        return analysis_image_path

def highest_N_value(array,number):
    #Returns the highest values
    """
        Private method that returns the N highest values
        
        Args:
        ----
        array(array): Matrix returned by function "calc_ICA"
        number(int)= Number of values to be returned
        -------
        Returns:
        result(array)=The N_highest values
        returnArgs(array)=The index of highest values
    """  
    #Return the largest n elements
    result=heapq.nlargest(number,array)
    resultArgs=map(array.tolist().index, heapq.nlargest(number,array))
    return result,resultArgs

def count_number_highest_value (matrix):
    
    #It counts all values greater than 50% of the highest value
    y=np.amax(matrix)*0.5
    count = sum(x > y for x in matrix )
    return count

def analysis_image(method,source): 
    """
        Private method that creates images of the most significant N channelss
        
        Args:
        ----
        
        Method(string)='Serpentine' or 'Patches'
        source(array)= Matrix returned by function "calc_ICA"
      
        
        
        Returns:
        Image
        -------
        Returns:
        Image
    """  
    #seletor select folder
    for seletor in range(1, 3):

        number_sources=source.shape[0]
        total_sum=np.zeros(165)
        for cont in range(number_sources):
            #Define soma 
            soma=np.zeros(165)

            #position in the vector desvector
            channel=np.array([])
            comparator=np.array(range(64))

            if (method=='Serpentine'):
                    #Use desvetor_serpentina in matshow
                    T=desvector_serpentina(comparator)

            if(method=='Patches'):
                    #Use desvetor_patches in matshow
                    T=desvector_patches(ordem,comparator)

            if seletor==1:
                N_highest_values = 3

            if seletor==2:
                N_highest_values = count_number_highest_value(source[cont])

            for num in range(N_highest_values):
                results=highest_N_value(source[cont],N_highest_values)


                #Create the image
                if (method=='Serpentine'):
                    #Use desvetor_serpentina in matshow
                    Matrix=desvector_serpentina(source[cont])
                    path=path_image_analysis_serpentina(number_sources,seletor)
                if(method=='Patches'):
                    #Use desvetor_patches in matshow
                    Matrix=desvector_patches(ordem,source[cont])
                    path=path_image_analysis_patches(number_sources,seletor)

                #position in the matrix 8X8
                position=np.argwhere(T==(results[1][num]))

                #Hottest pixel channel number
                channel=np.append(channel,(pixel_map[position[0][0],position[0][1]]))


            plt.figure(figsize=(15,10))
            plt.plot(data_array[:,7],label='Single-Anode PMT' )

            for num1 in range(N_highest_values):
                if channel[num1]!=0:
                    plt.plot(data_array[:,int((channel[num1]-1))],label='0'+str(num1+1)+' MA_PMT Channel: '+str(int(round(channel[num1]))))



            leg = plt.legend(bbox_to_anchor=(0.730, 0.999), loc="best", borderaxespad=0.,fontsize = 'x-large')
            leg.set_title("Plot Warmer Pixels", prop = {'size':'x-large'})
            leg._legend_box.align = "left"

            N=str(number_sources)
            X= str(cont+1)
            plt.title('0'+str(number_sources)+' Sources :'+' Analysis '+X+ ' FastICA Method '+method, fontsize = 20)
            plt.text(150, 1360, r'Ordem Decrescente')

            # hide tick and tick label of the big axes
            plt.xlabel("Triggers",fontsize=15)
            plt.ylabel("ADC Counts",fontsize=15)

            plt.savefig(path+'/analysis_'+N+'_image_'+X+'.png', bbox_inches='tight')



            plt.clf()
            #Plot Sum Pixels
            plt.figure(figsize=(15,10))
            plt.plot(data_array[:,7],label='Single-Anode PMT' )

            for num1 in range(N_highest_values):
                if channel[num1]!=0:
                    plt.plot(data_array[:,int((channel[num1]-1))],label='0'+str(num1+1)+' MA_PMT Channel: '+str(int(round(channel[num1]))))
                    soma=np.sum([soma,data_array[:,int((channel[num1]-1))]], axis=0)
            soma=soma/N_highest_values
            plt.plot(soma,label="Sum Pixels") 

            plt.title('0'+str(number_sources)+' Sources :'+' Sum Analysis '+X+ ' FastICA Method '+method , fontsize = 20)
            leg = plt.legend(bbox_to_anchor=(0.730, 0.999), loc="best", borderaxespad=0.,fontsize = 'x-large')
            leg.set_title("Plot Warmer Pixels", prop = {'size':'x-large'})
            leg._legend_box.align = "left"

            # hide tick and tick label of the big axes
            plt.xlabel("Triggers",fontsize=15)
            plt.ylabel("ADC Counts",fontsize=15)

            plt.savefig(path+'/Sum_analysis_'+N+'_image_'+X+'.png', bbox_inches='tight')



            plt.clf()
            #Todas as somas no mesmo grafico
            plt.figure(figsize=(15,10))

            plt.plot(data_array[:,7],label='Single-Anode PMT' )
            plt.plot(soma,label="Sum Pixels")  


            total_sum=np.vstack([total_sum,soma])

        for item in range(1,number_sources+1, 1):
            plt.plot(total_sum[item])


        # hide tick and tick label of the big axes
        plt.xlabel("Triggers",fontsize=15)
        plt.ylabel("ADC Counts",fontsize=15)


        plt.savefig(path+'/All_Sum_analysis.png', bbox_inches='tight')

print("Done-FastICA Images")
print("Running")
#Method Serpentine

analysis_image('Serpentine',sources03)

analysis_image('Serpentine',sources04)

analysis_image('Serpentine',sources05)

analysis_image('Serpentine',sources06)

analysis_image('Serpentine',sources07)

analysis_image('Serpentine',sources08)

analysis_image('Serpentine',sources09)

analysis_image('Serpentine',sources48)

analysis_image('Serpentine',sources64)


#Method Patches

analysis_image('Patches',psources03)

analysis_image('Patches',psources04)

analysis_image('Patches',psources05)

analysis_image('Patches',psources06)

analysis_image('Patches',psources07)

analysis_image('Patches',psources08)

analysis_image('Patches',psources09)

analysis_image('Patches',psources48)

analysis_image('Patches',psources64)


print("Done")

