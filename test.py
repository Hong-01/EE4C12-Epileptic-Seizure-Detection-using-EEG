def Classifier(clf,x_train,y_train,x_test,y_test):

    #fit the model
    clf.fit(x_train,y_train)

    #predict on the test set
    y_pred = clf.predict(x_test)

    accuracy_value=accuracy_score(y_test,y_pred)
    f1_value=f1_score(y_test,y_pred,average='macro')
    precision_value=precision_score(y_test,y_pred,average='macro')
    recall_value=recall_score(y_test,y_pred,average='macro')

    # print the scores
    print('Accuracy score: ', accuracy_value)
    print('F1 score: ', f1_value)
    print('Precision score: ', precision_value)
    print('Recall score: ', recall_value)
    
    # plot the confusion matrix

    plt.figure(figsize=(5,5))
    confusion=confusion_matrix(y_test,y_pred)
    sns.heatmap(confusion, annot=True, fmt='d',cmap="RdPu")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.show()
    # plt. ...
    # plt. ...
    plt.show()
    return accuracy_value,f1_value,precision_value,recall_value

def knn_classifier(X_train, y_train, X_test, y_test,neighbours):
    accuracy=[]
    f1=[]
    precision=[]
    recall=[]
    neighbours_list=[]
    time_list=[]
    for i in neighbours:
        start=time.time()
        knn = KNeighborsClassifier(n_neighbors=i)
        print('#############################################')
        print('knn with k =',i)
        print('#############################################')
        accuracy_value,f1_value,precision_value,recall_value=Classifier(knn,X_train,y_train,X_test,y_test)
        end=time.time()
        time_list.append(end-start)
        accuracy.append(accuracy_value)
        f1.append(f1_value)
        precision.append(precision_value)
        recall.append(recall_value)
        neighbours_list.append(i)

    accuracy_max=max(accuracy)
    accuracy_max_index=accuracy.index(accuracy_max)
    time_accuracy_list=[]

    if accuracy.count(accuracy_max)>1:
        for i in np.where(accuracy==accuracy_max)[0]:
            time_accuracy_list.append((1/time_list[i])+accuracy[i])
        accuracy_max_index=time_accuracy_list.index(max(time_accuracy_list))
        accuracy_max_index=np.where(accuracy==accuracy_max)[0][accuracy_max_index]
        accuracy_max=accuracy[accuracy_max_index]
        time_min=time_list[accuracy_max_index]
    
    if accuracy.count(accuracy_max)==1:
        time_min=time_list[accuracy_max_index]

    best_k=neighbours_list[accuracy_max_index]

    # print("*************************************")
    # print('Best k is',best_k)
    # print("Highest accuracy is",accuracy_max)
    # print("*************************************")

    return accuracy,f1,precision,recall,accuracy_max,best_k,time_min
        
    

def svc_classifier( X_train, y_train, X_test, y_test,kernel,C=[1],max_iter=[-1]):
    accuracy=[]
    f1=[]
    precision=[]
    recall=[]
    kernel_list=[]
    C_list=[]
    max_iter_list=[]
    time_list=[]

    for i in kernel:
        for j in C:
            for k in max_iter:
                start=time.time()
                svm = SVC(kernel=i,gamma='auto', random_state=42, C=j, coef0=0.0, tol=1e-3, max_iter=k)
                print('#############################################')
                print('svm with kernel of',i)
                print('C =',j)
                print('max_iter =',k)
                print('#############################################')
                accuracy_value,f1_value,precision_value,recall_value=Classifier(svm,X_train,y_train,X_test,y_test)
                end=time.time()
                time_list.append(end-start)
                accuracy.append(accuracy_value)
                f1.append(f1_value)
                precision.append(precision_value)
                recall.append(recall_value)
                kernel_list.append(i)
                C_list.append(j)
                max_iter_list.append(k)


    
    accuracy_max=max(accuracy)
    accuracy_max_index=accuracy.index(accuracy_max)
    time_accuracy_list=[]

    if accuracy.count(accuracy_max)>1:
        for i in np.where(accuracy==accuracy_max)[0]:
            time_accuracy_list.append((1/time_list[i])+accuracy[i])
        accuracy_max_index=time_accuracy_list.index(max(time_accuracy_list))
        accuracy_max_index=np.where(accuracy==accuracy_max)[0][accuracy_max_index]
        accuracy_max=accuracy[accuracy_max_index]
        time_min=time_list[accuracy_max_index]
    if accuracy.count(accuracy_max)==1:
        time_min=time_list[accuracy_max_index]
    best_kernel=kernel_list[accuracy_max_index]
    best_C=C_list[accuracy_max_index]
    best_max_iter=max_iter_list[accuracy_max_index]
    # print("*************************************")
    # print('Best kernel is',best_kernel)
    # print('Best C is',best_C)
    # print('Best max_iter is',best_max_iter)
    # print("Highest accuracy is",accuracy_max)
    # print("*************************************")



    return accuracy,f1,precision,recall,accuracy_max,best_kernel,best_C,best_max_iter,time_min



def mlp_classifier( X_train, y_train, X_test, y_test,hidden_layer_sizes,alpha=[0.0001],max_iter=[200]):
    accuracy=[]
    f1=[]
    precision=[]
    recall=[]
    hidden_layer_sizes_list=[]
    alpha_list=[]
    max_iter_list=[]
    time_list=[]
    for i in hidden_layer_sizes:
        for j in alpha:
            for k in max_iter:
                start=time.time()
                mlp = MLPClassifier(hidden_layer_sizes=i,activation='relu',solver='adam',alpha=j,max_iter=k,random_state=42)
                print('#############################################')
                print('mlp with hidden layer of',i)
                print('alpha =',j)
                print('max_iter =',k)
                print('#############################################')
                accuracy_value,f1_value,precision_value,recall_value=Classifier(mlp,X_train,y_train,X_test,y_test)
                end=time.time()
                time_list.append(end-start)
                accuracy.append(accuracy_value)
                f1.append(f1_value)
                precision.append(precision_value)
                recall.append(recall_value)
                hidden_layer_sizes_list.append(i)
                alpha_list.append(j)
                max_iter_list.append(k)
                
    accuracy_max=max(accuracy)
    accuracy_max_index=accuracy.index(accuracy_max)
    time_accuracy_list=[]

    if accuracy.count(accuracy_max)>1:
        for i in np.where(accuracy==accuracy_max)[0]:
            time_accuracy_list.append((1/time_list[i])+accuracy[i])
        accuracy_max_index=time_accuracy_list.index(max(time_accuracy_list))
        accuracy_max_index=np.where(accuracy==accuracy_max)[0][accuracy_max_index]
        accuracy_max=accuracy[accuracy_max_index]
        time_min=time_list[accuracy_max_index]
    if accuracy.count(accuracy_max)==1:
            time_min=time_list[accuracy_max_index]

    best_hidden_layer_sizes=hidden_layer_sizes_list[accuracy_max_index]
    best_alpha=alpha_list[accuracy_max_index]
    best_max_iter=max_iter_list[accuracy_max_index]
    # print("*************************************")

    # print('Best hidden layer sizes is',best_hidden_layer_sizes)
    # print('Best alpha is',best_alpha)
    # print('Best max_iter is',best_max_iter)
    # print("Highest accuracy is",accuracy_max)
    # print("*************************************")

    return accuracy,f1,precision,recall,accuracy_max,best_hidden_layer_sizes,best_alpha,best_max_iter,  time_min


def Model_train(X_train, X_test, y_train, y_test, model_choice=['KNN', 'SVC','MLP'], knn_property={}, svc_property={}, mlp_property={}):
    """Example:best_model_type,best_model_property,accuracy_max,time_min=Model_train(X_train, X_test, y_train, y_test, 
            model_choice=['KNN', 'SVC','MLP'], 
            knn_property={'neighbours':[3,4,5]}, 
            svc_property={'kernal':['linear', 'poly', 'rbf'],'C':[1],'max_iter':[-1]}, 
            mlp_property={'hidden_layer_sizes':[(100),(100,100),(100,100,100)],'alpha':[0.0001],'max_iter':[200]})"""
    accuracy=[]
    model_type=[]
    model_property=[]
    time_list=[]

    if 'KNN' in model_choice:
        knn_accuracy,knn_f1,knn_precision,knn_recall,knn_accuracy_max,knn_best_k,time_min=knn_classifier(
             X_train, y_train, X_test, y_test,neighbours=knn_property['neighbours'])
        accuracy.append(knn_accuracy_max)
        model_type.append('KNN')
        model_property.append(["neighbour= "+ str(knn_best_k)])
        time_list.append(time_min)

    if 'SVC' in model_choice:
        svc_accuracy,svc_f1,svc_precision,svc_recall,svc_accuracy_max,svc_best_kernel,svc_best_C,svc_best_max_iter,time_min=svc_classifier(
             X_train, y_train, X_test, y_test,kernel=svc_property['kernal'],C=svc_property['C'],max_iter=svc_property['max_iter'])
        accuracy.append(svc_accuracy_max)
        model_type.append('SVC')
        model_property.append(["kernel= "+ str(svc_best_kernel),"C= "+ str(svc_best_C),"max_iter= "+ str(svc_best_max_iter)])
        time_list.append(time_min)

    if 'MLP' in model_choice:
        mlp_accuracy,mlp_f1,mlp_precision,mlp_recall,mlp_accuracy_max,mlp_best_hidden_layer_sizes,mlp_best_alpha,mlp_best_max_iter,time_min=mlp_classifier(
             X_train, y_train, X_test, y_test, hidden_layer_sizes=mlp_property['hidden_layer_sizes'],alpha=mlp_property['alpha'],max_iter=mlp_property['max_iter'])
        accuracy.append(mlp_accuracy_max)
        model_type.append('MLP')
        model_property.append(["hidden_layer_sizes= "+ str(mlp_best_hidden_layer_sizes),"alpha= "+ str(mlp_best_alpha),"max_iter= "+ str(mlp_best_max_iter)])
        time_list.append(time_min)

    accuracy_max=max(accuracy)
    accuracy_max_index=accuracy.index(accuracy_max)

    time_accuracy_list=[]
    if accuracy.count(accuracy_max)>1:
        for i in np.where(accuracy==accuracy_max)[0]:
            time_accuracy_list.append((1/time_list[i])+accuracy[i])
        accuracy_max_index=time_accuracy_list.index(max(time_accuracy_list))
        accuracy_max_index=np.where(accuracy==accuracy_max)[0][accuracy_max_index]
        accuracy_max=accuracy[accuracy_max_index]
        time_min=time_list[accuracy_max_index]
    if accuracy.count(accuracy_max)==1:
            time_min=time_list[accuracy_max_index]
    
    best_model_type=model_type[accuracy_max_index]
    best_model_property=model_property[accuracy_max_index]
    print('')
    print('')
    print("*************************************")
    print('Best model type is',best_model_type)
    print('Best model property is',best_model_property)
    print("Highest accuracy is",accuracy_max)
    print("Lowest time (s) is",time_min)
    print("*************************************")

    
    return best_model_type,best_model_property,accuracy_max,time_min