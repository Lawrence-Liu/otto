set.seed(111)
data<-read.csv("train.csv",header=T)
data<-data[,-1]
group_data<-split(data[,-94],data$target)
cent<-lapply(group_data,colMeans)
cent<-matrix(unlist(cent),9,93,T)
distance<-dist(cent)
dist<-matrix(0,9,9)
k=1
for(i in 1:8){
  for(j in (i+1):9){
    dist[j,i]<-distance[k]
    dist[i,j]<-dist[j,i]
    k=k+1
  }
}
colnames(dist)<-c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
rownames(dist)<-colnames(dist)
#see the distance of centers of each cluster
dist_heatmap<-heatmap(dist,Rowv=NA,Colv=NA,col=heat.colors(10),scale="column",margins=c(10,10),main="Euclidean distance between each cluster centre")

#ordinary pca
pca<-prcomp(~.,data[,-94])
pca_data<-data.frame(pca$x)
pca_group<-split(pca_data,data$target)
subindex<-sample(1:61878,800)
sub_data<-data[subindex,]
sub_pca_data<-pca_data[subindex,]
sub_pca_group<-split(sub_pca_data,data$target[subindex])
plot(sub_pca_group$Class_1[,c(1,2)],col="red",xlim=c(-30,20),ylim=c(-30,20),main="2-D Projection plot of 800 sample points.")
points(sub_pca_group$Class_2[,c(1,2)],pch=3,col="yellow")
points(sub_pca_group$Class_3[,c(1,2)],pch=2,col="blue")
points(sub_pca_group$Class_4[,c(1,2)],pch=20,col="grey")
points(sub_pca_group$Class_5[,c(1,2)],pch="o",col="black")
points(sub_pca_group$Class_6[,c(1,2)],pch="*",col="brown")
points(sub_pca_group$Class_7[,c(1,2)],pch=11,col="purple")
points(sub_pca_group$Class_8[,c(1,2)],pch=13,col="orange")
points(sub_pca_group$Class_9[,c(1,2)],pch=9,col="lavender")
legend(8,20,legend=colnames(dist),pch=c(1,3,2,20,111,42,11,13,9),col=c("yellow","blue","grey","black","brown","purple","orange","lavender"))