

colscale = scale_color_manual(values=c("#7CB637", "#666666", "#4381C1")) 
colscale_fill = scale_fill_manual(values=c("#7CB637", "#666666", "#4381C1")) 

remove_quotes = function(d) {
  if (!is.null(d$modal)) {
    d$modal = gsub('"', '', d$modal)
  }
  d$color = gsub('"', '', d$color)
  d$speaker_type = gsub('"', '', d$speaker_type)
  
  return(d)
}

format_data = function(d) {
  
  drops <- c("modal1","rating1")
  d2 = d[ , !(names(d) %in% drops)]
  setnames(d2, old=c("rating2","modal2"), new=c("rating", "modal"))
  
  drops <- c("modal2","rating2")
  d3 = d[ , !(names(d) %in% drops)]
  setnames(d3, old=c("rating1","modal1"), new=c("rating", "modal"))
  
  drops <- c("modal2", "rating2", "modal1", "rating1")
  d4 = d[ , !(names(d) %in% drops)]
  d4$rating = d4$rating_other
  d4$modal = "other"
  
  d = rbind(d2, d3, d4)
  
  d$modal = gsub('"', '', d$modal)
  d$modal = factor(d$modal, levels = c("might", "probably", "other"), ordered = T)
  
  d$percentage_blue_f = factor(d$percentage_blue)
  d_blue = d %>% filter(., grepl("blue", sentence2))
  d_orange = d %>% filter(., grepl("orange", sentence2))
  
  d_orange_reverse = d_orange
  d_orange_reverse$percentage_blue = 100-d_orange$percentage_blue
  
  d_comparison = rbind(d_blue, d_orange_reverse)
  d_comparison$blue= grepl("blue", d_comparison$sentence2)
  d_comparison$percentage_blue_f = factor(d_comparison$percentage_blue)
  
  return(d_comparison)
  
}

auc_method2 = function(d) {
  auc = AUC(x=d$percentage_blue, y=d$rating_m, method="spline")
  return(auc)
}


auc_for_participants = function(d, method) {
  
  d[d$color=="orange",]$percentage_blue = 100 - d[d$color=="orange",]$percentage_blue 
  
  aucs = d %>% 
    group_by(workerid) %>% 
    summarize(.groups = "drop_last", test_order = first(test_order),
              first_speaker_type = first(first_speaker_type),
              confident_speaker = first(confident_speaker))
  
  
  aucs$auc_might = 0
  aucs$auc_probably = 0
  
  
  i = 1
  
  for (wid in unique(d$workerid)) {
    d.might_ratings = d %>% 
      filter (workerid == wid) %>%
      filter (modal == "might") %>%
      group_by(workerid, percentage_blue) %>%
      summarise(rating_m = mean(rating))
    
    aucs$auc_might[i] = method(d.might_ratings)
    
    d.probably_ratings = d %>% 
      filter (workerid == wid) %>%
      filter (modal == "probably") %>%
      group_by(workerid, percentage_blue) %>%
      summarise(rating_m = mean(rating))
    
    aucs$auc_probably[i] = method(d.probably_ratings)
    
    i = i + 1
  }
  
  aucs$auc_diff = aucs$auc_might - aucs$auc_probably
  
  return(aucs)
}


myNorm = function(x) {
  xmin = min(x, na.rm = TRUE)
  xmax = max(x, na.rm = TRUE) 
  x_scaled = (x - xmin)/(xmax - xmin) * 2 -1
  return(x_scaled)
}

myCenter <- function(x) {
  if (is.numeric(x)) { return(x - mean(x, na.rm=T)) }
  if (is.factor(x)) {
    x <- as.numeric(x)
    return(x - mean(x))
  }
  if (is.data.frame(x) || is.matrix(x)) {
    m <- matrix(nrow=nrow(x), ncol=ncol(x))
    colnames(m) <- paste("c", colnames(x), sep="")
    for (i in 1:ncol(x)) {
      if (is.factor(x[,i])) {
        y <- as.numeric(x[,i])
        m[,i] <- y - mean(y, na.rm=T)
      }
      if (is.numeric(x[,i])) {
        m[,i] <- x[,i] - mean(x[,i], na.rm=T)
      }
    }
    return(as.data.frame(m))
  }
}

library(bootstrap)
theta <- function(x,xdata,na.rm=T) {mean(xdata[x],na.rm=na.rm)}
ci.low <- function(x,na.rm=T) {
  mean(x,na.rm=na.rm) - quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.025,na.rm=na.rm)}
ci.high <- function(x,na.rm=T) {
  quantile(bootstrap(1:length(x),1000,theta,x,na.rm=na.rm)$thetastar,.975,na.rm=na.rm) - mean(x,na.rm=na.rm)}

