class item_list:
    def __init__(self, word, word_count=0, parent=None, link=None):
        self.word=word
        self.word_count=word_count
        self.parent=parent
        self.link=link
        self.child={}

#tree traversal
    def list_tree(self):
        output=[]
        output.append(str(self.word) + " " +str(self.word_count))
        if len(list(self.child.keys()))>0:
            for i in (list(self.child.keys())):
                #print("self",self.child.keys(),self.parent)
                output.append(self.child[i].list_tree())
        return output
  
              
'''      Build FPTREE class and method       '''        
class fp_growth_tree:
    def __init__(self, data, min_support=400):
        self.data=data
        self.min_support=min_support
        
        self.root= item_list(word="Null", word_count=1)
        self.word_line_sort=[]
        self.updated_table=[]
        self.word_sort_dic=[]
        
        # Contains number of times the given items was present in all the transactions
        self.word_dic={}
        
        self.word_order_dic={}
        self.construct(data)
    def construct(self, data):
        #print("construct 1",data)
        
        # Extract items in the given transactions
        for tran in data:
            for words in tran:
                #print("words",words)
                if words in self.word_dic.keys():
                    self.word_dic[words]+=1
                else:
                    self.word_dic[words]=1
        
        list_of_word = list(self.word_dic.keys())
        #print("list of word",list_of_word,self.word_dic['I2'])
        
        # Delete the items whose count < min support
        for word in list_of_word:
            if(self.word_dic[word]<self.min_support):
                del self.word_dic[word]
        
        #Sort the items based on their frequency
        self.word_sort_dic = sorted(self.word_dic.items(), key=lambda x: (-x[1],x[0])) 
        
        #Create a table with Sno, item, its frequency count
        t=0
        for i in self.word_sort_dic:
            word = i[0]
            wordc = i[1]
            self.word_order_dic[word]=t
            t+=1
            word_info = {'wordn':word, 'wordcc':wordc, 'linknode': None}
            self.updated_table.append(word_info)
        #Process each transaction records
        for line in data:
            #print("line",line)
            
            # Remove the unwanted items in given transaction set
            supported_words=[]
            for word in line:
                if word in self.word_dic.keys():
                    supported_words.append(word)
            #print("supported words",supported_words)
            
            if len(supported_words)>0:
                
                #Sort the items in the given transactions
                sortsupword = sorted(supported_words, key = lambda k: self.word_order_dic[k])
                #print("sortsupword",sortsupword)
                self.word_line_sort.append(sortsupword)
                element = self.root
                
                #Start from root of the tree
                for i in sortsupword:                  
                    #Checks if already item is there, increment its word count.
                    if i in element.child.keys():
                        element.child[i].word_count +=1
                        #print("ifblock",element.child[i].word)
                        element=element.child[i]
                        #print("after child if block",element.word)
                    #If element is not created, create a new one. It can be during start up (when a branch creation) 
                    #or when new item starts from root than previous one
                    else:
                        #print("else  block",i, element.child.keys())
                        element.child[i] = item_list(word=i,word_count=1,parent=element,link=None)
                        element=element.child[i]
                        #Update the table with link node details if created.
                        for word_info in self.updated_table:
                            #print("word info inside for loop",word_info)
                            if word_info["wordn"] == element.word:
                                #This updates if it is a single branch. For multiple branchs go to else block
                                if word_info["linknode"] is None:
                                    #print("inside for if block",word_info['linknode'],element)
                                    word_info["linknode"] = element
                                
                                #Form a single link list to identify if there are multiple branchs exists.    
                                else:
                                    iter_node = word_info["linknode"]
                                    #print("word info inside for loop else block 1",iter_node,iter_node.link,iter_node.word)
                                    while(iter_node.link is not None):
                                        iter_node = iter_node.link
                                    iter_node.link = element
                                    #print("word info inside for loop else block 2",iter_node,iter_node.link)
                                    

    def condition_tree_transaction(self,element):
        if element.parent is None:
            return None
        #print("start",element.word)
        condtreeline =[]
        while element is not None:
            line=[]
            #print("first while loop",element.word)
            parent_element = element.parent
            while parent_element.parent is not None:
                line.append(parent_element.word)
                parent_element=parent_element.parent
            line = line[::-1]
            #print("line",line)
            for i in range(element.word_count):
                #print("i",i,condtreeline)
                condtreeline.append(line)   
            #print("while loop",element.word)
            element=element.link
        #print("condtreeline",condtreeline)
        return condtreeline


    def condition_tree_transaction_top_down(self,element):
        #print("inside condition tree top down",element)
        if(element is None):
            return None
        if len(element.child.keys()) == 0:
            if(element.link is None):
                return None
            else:
                element = element.link
                #print("else bolock", element.word)
        #print("after if block",element.child.keys(),element.word)
        condtreeline =[]
        #Start from element of the tree
        line=[]
        for i in element.child.keys():
            #print("i value",i,element.child[i].word)
            #Checks if already item is there, increment its word count.
            #line.append(element.child[i].word)
            line.append(str(element.child[i].word) + ":" + str(element.child[i].word_count))
            line1=(self.condition_tree_transaction_top_down(element.child[i]))
            #print("line1",line1)
            if(line1 is not None):
                line.append(line1)
            #print("after child if block",element.word,line1)
            #print("line",line)
        if(line == []):
            return None
            
        line = line[::-1]
        condtreeline.append(line)   
        #print("condtreeline",condtreeline)

        return line
    
    def frequent_word_list(self,parentnode=None):
        if len(list(self.root.child.keys()))==0:
            return None
        #print("1",parentnode)
        result=[]
        sup=self.min_support
        revtable = self.updated_table[::-1]
        #print("revtable",revtable)
        for n in revtable:
            #print("n",n)
            fqset=[set(),0]
            if(parentnode==None):      
                fqset[0]={n['wordn'],}
                #print("parentnode==None",fqset[0])
            else:
                fqset[0] = {n['wordn']}.union(parentnode[0])
                #print("else block",fqset[0])
                
            fqset[1]=n['wordcc']
            result.append(fqset)
            condtran = self.condition_tree_transaction(n['linknode'])
            contree= fp_growth_tree(condtran,sup)
            conwords = contree.frequent_word_list(fqset)
            if conwords is not None:
                for words in conwords:
                    result.append(words)
            
        return result

    def checkheight(self):
        if len(list(self.root.child.keys()))==0:
            return False
        else:
            return True
          
          
min_sup=2


'''
test_data = [['I1','I2','I5'],
             ['I2','I4'],
             ['I2','I3'],
             ['I1','I2','I4'],
             ['I1','I3'],
             ['I2','I3'],
             ['I1','I3'],
             ['I1','I2','I3','I5'],
             ['I1','I2','I3']]


test_data = [['I1','I2','I5'],
             ['I2','I3','I4'],
             ['I3','I4'],
             ['I1','I2','I3','I4']]

'''
test_data = [['A','B'],
             ['B','C','D'],
             ['A','C','D','E'],
             ['A','D','E'],
             ['A','B','C']]
'''
test_data = [['I1','I2','I5'],
            ['I2','I3','I4'],
            ['I3','I4'],
            ['I1','I2','I3','I4']]

'''
fp_tree = fp_growth_tree(test_data, min_sup) #create FP tree on data


frequentwordset = fp_tree.frequent_word_list() #mining frequent patt

frequentwordset=sorted(frequentwordset,key = lambda k: -k[1] )
print(frequentwordset)


for word in frequentwordset:
    count = (str(word[1])+"\t")
    words =''
    for val in word[0]:
        words+= (str(word[0])+" ")


print("Bottom up approach")
for i in fp_tree.updated_table[::-1]:
    lines = fp_tree.condition_tree_transaction(i['linknode'])
    #print("lines =============>",i['wordn'],lines)
    condtree = fp_growth_tree(lines,min_sup)
    if(condtree.checkheight()):
        print('Condtional FPTree Root on '+(i['wordn']))
        print(condtree.root.list_tree())

print("Top Down approach")
for i in fp_tree.updated_table[::-1]:
    lines = fp_tree.condition_tree_transaction_top_down(i['linknode'])
    print("Condtional FPTree Root on",i['wordn'],lines)
