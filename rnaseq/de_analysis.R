#!/usr/bin/env Rscript

# Install required packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
#BiocManager::install("DESeq2")
#BiocManager::install(c("DESeq2", "ggplot2"), update = FALSE)
#BiocManager::install(c("DESeq2", "ggplot2", "EnhancedVolcano"), update = FALSE)

# Load required libraries
library(DESeq2)
library(ggplot2)
library(EnhancedVolcano)

# Load count data
counts <- read.table("E:\\bioinfoo\\bioinfo_pour_o\\counts\\counts.txt", header=TRUE, row.names=1)
# Extract count data (columns 6 to 17)
count_data <- counts[,6:17]

# Clean up column names
colnames(count_data) <- gsub("^.*\\.(.+)_sorted\\.bam$", "\\1", colnames(count_data))

# Create a vector of condition labels
# Adjust this based on your experimental design
# Ensure this matches the order of your samples in the count matrix
condition <- factor(c("A", "B", "C", "D", "A", "A", "B", "B", "C", "C", "D", "D"))

# Create DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = count_data,
                              colData = data.frame(condition),
                              design = ~ condition)
# Run DESeq2
dds <- DESeq(dds)
res <- results(dds)

# EnhancedVolcano plot
png("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\enhanced_volcano_plot_all.png", width=1200, height=1000, res=150)
print(EnhancedVolcano(res,
    lab = rownames(res),
    x = 'log2FoldChange',
    y = 'pvalue',
    title = 'Differential Expression',
    pCutoff = 0.05,
    FCcutoff = 1,
    pointSize = 3.0,
    labSize = 6.0,
    col = c('black', 'black', 'black', 'red3'),
    colAlpha = 1,
    legendPosition = 'right',
    legendLabSize = 10,
    legendIconSize = 3.0,
    drawConnectors = TRUE,
    widthConnectors = 0.75
))
dev.off()

# MA plot
png("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\ma_plot_all.png", width=800, height=600)
plotMA(res, main="MA Plot")
dev.off()

# PCA plot
vsd <- vst(dds, blind=FALSE)
png("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\pca_plot_all_pointsonly.png", width=800, height=600)
plotPCA(vsd, intgroup="condition")
dev.off()


####PCA plot for paper####
vsd <- vst(dds, blind=FALSE)
pca_plot <- plotPCA(vsd, intgroup="condition", returnData=TRUE)
pca_plot$group <- factor(pca_plot$group, levels=c("A", "B", "C", "D"))

# Create custom PCA plot
# Define explicit colors from Set2 palette
# Define explicit colors and shapes for each condition
set2_colors <- c("A" = "#66C2A5", "B" = "#FC8D62", "C" = "#8DA0CB", "D" = "#E78AC3")
condition_shapes <- c("A" = 16, "B" = 17, "C" = 18, "D" = 15)

p <- ggplot(pca_plot, aes(PC1, PC2, color = group, shape = group)) +
  geom_point(size = 4) +
  scale_color_manual(values = set2_colors, 
                     labels = c("A" = "agar:agar", "B" = "agar:scaffold", 
                                "C" = "scaffold:scaffold", "D" = "scaffold:agar")) +
  scale_shape_manual(values = condition_shapes,
                     labels = c("A" = "agar:agar", "B" = "agar:scaffold", 
                                "C" = "scaffold:scaffold", "D" = "scaffold:agar")) +
  theme_minimal() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_line(color = "gray95", size = 0.1),
    axis.line = element_line(color = "black"),
    legend.position = c(0.99, 0.05),  # Position the legend at the bottom right
    legend.justification = c(1, 0),  # Align the legend to the bottom right
    legend.background = element_rect(fill = "white", color = "#575757", size = 0.5),  # Changed border color to a paler gray
    legend.margin = margin(5, 5, 3, 5),  # Add some margin inside the legend box
    aspect.ratio = 0.6,  # This makes the plot more elongated
    axis.title = element_text(size = 14),  # Increase the size of x and y labels
    axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 0, l = 0)),  # Move x-axis title further away
    axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),  # Move y-axis title further away
    legend.title = element_text(size = 9)  # Make the legend title a bit smaller
  ) +
  labs(x = paste0("PC1: ", round(attr(pca_plot, "percentVar")[1] * 100, 2), "% of variance"), 
       y = paste0("PC2: ", round(attr(pca_plot, "percentVar")[2] * 100, 2), "% of variance"),
       color = "Ancestry habitat : Growing habitat", 
       shape = "Ancestry habitat : Growing habitat") +
  coord_fixed(ratio = 0.6)  # This also contributes to making the plot more elongated

# Save the plot
ggsave("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\pca_plot_forpaper.jpg", 
       p, 
       dpi=300, 
       width = 8,  # Adjust as needed
       height = 5,  # Adjust as needed
       units = "in")




# Custom PCA plot with group ellipses
vsd <- vst(dds, blind=FALSE)

# Create PCA plot
pca_plot <- plotPCA(vsd, intgroup="condition", returnData=TRUE)

# Create the base plot
p <- ggplot(pca_plot, aes(PC1, PC2, color=condition)) +
  geom_point(size=3) +
  geom_text(aes(label=name), vjust=2, hjust=0.5) +
  theme_bw() +
  ggtitle("PCA Plot")

# Add ellipses using geom_polygon with adjusted confidence level
ellipses <- data.frame()
for(cond in unique(pca_plot$condition)) {
  sub_data <- subset(pca_plot, condition == cond)
  # Reduce confidence level to 68% (roughly 1 standard deviation)
  ell <- dataEllipse(sub_data$PC1, sub_data$PC2, levels=0.68, draw=FALSE)
  ell_df <- data.frame(ell)
  ell_df$condition <- cond
  ellipses <- rbind(ellipses, ell_df)
}

p <- p + geom_polygon(data=ellipses, aes(x=x, y=y, fill=condition), alpha=0.2, inherit.aes=FALSE)

# Optional: Add convex hulls instead of ellipses
p <- p + stat_chull(geom = "polygon", alpha = 0.1, aes(fill = condition))

# Adjust the plot limits to focus on the data points
p <- p + coord_cartesian(xlim = c(min(pca_plot$PC1) * 1.1, max(pca_plot$PC1) * 1.1),
                         ylim = c(min(pca_plot$PC2) * 1.1, max(pca_plot$PC2) * 1.1))

# Save the plot
png("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\pca_plot_all_improved.png", width=800, height=600)
print(p)
dev.off()


# Simply enclose the points in ellipses
# Create PCA plot
vsd <- vst(dds, blind=FALSE)

# Create PCA plot
pca_plot <- plotPCA(vsd, intgroup="condition", returnData=TRUE)

# Function to create hull data
create_hull <- function(df) {
  hull <- df[chull(df$PC1, df$PC2), ]
  hull <- rbind(hull, hull[1, ])  # Close the polygon
  return(hull)
}

# Create hull data for each condition
hull_data <- do.call(rbind, lapply(split(pca_plot, pca_plot$condition), create_hull))

# Create the plot
p <- ggplot(pca_plot, aes(PC1, PC2, color=condition)) +
  geom_point(size=3) +
  geom_text(aes(label=name), vjust=2, hjust=0.5) +
  geom_polygon(data=hull_data, aes(x=PC1, y=PC2, fill=condition), alpha=0.1, inherit.aes=FALSE) +
  theme_bw() +
  ggtitle("PCA Plot")

# Adjust the plot limits to focus on the data points
p <- p + coord_cartesian(xlim = c(min(pca_plot$PC1) * 1.1, max(pca_plot$PC1) * 1.1),
                         ylim = c(min(pca_plot$PC2) * 1.1, max(pca_plot$PC2) * 1.1))

# Customize the legend
p <- p + theme(legend.position = "right") +
         scale_color_discrete(name = "Condition") +
         scale_fill_discrete(name = "Condition")

# Save the plot
png("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\pca_plot_convex_hull.png", width=800, height=600)
print(p)
dev.off()


# Save results
write.csv(as.data.frame(res), "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\differential_expression_results_all.csv")

print("Analysis complete. Check the current directory for output files.")




####Pairwise analysis####

perform_pairwise_comparison <- function(dds, group1, group2) {
    # Determine reference group
    ref_groups <- c("A", "C", "B")
    ref_group <- ref_groups[ref_groups %in% c(group1, group2)][1]
    
    # Ensure ref_group is always the reference
    if (group1 == ref_group) {
        temp <- group1
        group1 <- group2
        group2 <- temp
    }
    
    # Subset the DESeqDataSet for the two groups
    dds_subset <- dds[, dds$condition %in% c(group1, group2)]
    dds_subset$condition <- droplevels(dds_subset$condition)
    dds_subset$condition <- relevel(dds_subset$condition, ref = ref_group)
    
    # Run DESeq2
    dds_subset <- DESeq(dds_subset)
    res <- results(dds_subset, contrast = c("condition", group1, group2))
    
    # Calculate overall min/max for x and y axes (if not pre-computed)
    if (!exists("global_limits")) {
        all_results <- combn(levels(dds$condition), 2, function(pair) {
            dds_temp <- dds[, dds$condition %in% pair]
            dds_temp$condition <- droplevels(dds_temp$condition)
            dds_temp <- DESeq(dds_temp)
            results(dds_temp, contrast = c("condition", pair[1], pair[2]))
        }, simplify = FALSE)
        
        global_limits <- list(
            x_min = min(sapply(all_results, function(r) min(r$log2FoldChange, na.rm = TRUE))),
            x_max = max(sapply(all_results, function(r) max(r$log2FoldChange, na.rm = TRUE))),
            y_min = 0,
            y_max = max(sapply(all_results, function(r) max(-log10(r$pvalue), na.rm = TRUE)))
        )
    }
    
    # EnhancedVolcano plot
    png(sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\enhanced_volcano_%s_vs_%s.png", group1, group2), width=1200, height=1000, res=150)
    print(EnhancedVolcano(res,
        lab = rownames(res),
        x = 'log2FoldChange',
        y = 'pvalue',
        title = sprintf('Differential Expression: %s vs %s', group1, group2),
        pCutoff = 0.05,
        FCcutoff = 1,
        pointSize = 3.0,
        labSize = 6.0,
        col = c('black', 'black', 'black', 'red3'),
        colAlpha = 1,
        legendPosition = 'right',
        legendLabSize = 10,
        legendIconSize = 3.0,
        drawConnectors = TRUE,
        widthConnectors = 0.75,
        xlim = c(global_limits$x_min, global_limits$x_max),
        ylim = c(global_limits$y_min, global_limits$y_max)
    ))
    dev.off()
    
    # MA plot
    png(sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\ma_plot_%s_vs_%s.png", group1, group2), width=800, height=600)
    plotMA(res, main=sprintf("MA Plot: %s vs %s", group1, group2), ylim=c(global_limits$x_min, global_limits$x_max))
    dev.off()
    
    # Save results
    write.csv(as.data.frame(res), sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\differential_expression_results_%s_vs_%s.csv", group1, group2))
}

# Perform all possible pairwise comparisons
groups <- levels(dds$condition)
combn(groups, 2, function(pair) perform_pairwise_comparison(dds, pair[1], pair[2]), simplify = FALSE)



####Group pair analysis####

# Function to perform comparison between pairs of groups
perform_group_pair_comparison <- function(dds, group_pair1, group_pair2) {
    # Create new condition labels
    new_condition <- ifelse(dds$condition %in% group_pair1, paste(group_pair1, collapse = ""), 
                     ifelse(dds$condition %in% group_pair2, paste(group_pair2, collapse = ""), NA))
    
    # Print debugging information
    cat("Group pair 1:", paste(group_pair1, collapse = ", "), "\n")
    cat("Group pair 2:", paste(group_pair2, collapse = ", "), "\n")
    cat("Unique new conditions:", paste(unique(new_condition), collapse = ", "), "\n")
    
    # Check if there are samples for both group pairs
    if (sum(!is.na(new_condition)) == 0) {
        cat("Error: No samples found for either group pair. Skipping this comparison.\n\n")
        return(NULL)
    }
    
    # Subset the DESeqDataSet for the two group pairs
    dds_subset <- dds[, !is.na(new_condition)]
    dds_subset$new_condition <- factor(new_condition[!is.na(new_condition)])
    
    # Print more debugging information
    cat("Number of samples in subset:", ncol(dds_subset), "\n")
    cat("Levels of new_condition:", paste(levels(dds_subset$new_condition), collapse = ", "), "\n\n")
    
    # Ensure there are exactly two levels
    if (length(levels(dds_subset$new_condition)) != 2) {
        cat("Error: There must be exactly two group levels for comparison. Skipping this comparison.\n\n")
        return(NULL)
    }
    
    # Set the reference level to the group containing "A"
    ref_level <- ifelse("A" %in% group_pair1, paste(group_pair1, collapse = ""), paste(group_pair2, collapse = ""))
    dds_subset$new_condition <- relevel(dds_subset$new_condition, ref = ref_level)
    
    # Run DESeq2
    design(dds_subset) <- ~new_condition
    dds_subset <- DESeq(dds_subset)
    
    # Get the comparison level (non-reference level)
    comp_level <- levels(dds_subset$new_condition)[levels(dds_subset$new_condition) != ref_level]
    
    res <- results(dds_subset, contrast = c("new_condition", comp_level, ref_level))
    
    # Create group pair names for file naming
    ref_group_name <- ref_level
    comp_group_name <- comp_level
    
    # EnhancedVolcano plot
    png(sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\enhanced_volcano_%s_vs_%s.png", comp_group_name, ref_group_name), width=1200, height=1000, res=150)
    print(EnhancedVolcano(res,
        lab = rownames(res),
        x = 'log2FoldChange',
        y = 'pvalue',
        title = sprintf('Differential Expression: %s vs %s', comp_group_name, ref_group_name),
        pCutoff = 0.05,
        FCcutoff = 1,
        pointSize = 3.0,
        labSize = 6.0,
        col = c('black', 'black', 'black', 'red3'),
        colAlpha = 1,
        legendPosition = 'right',
        legendLabSize = 10,
        legendIconSize = 3.0,
        drawConnectors = TRUE,
        widthConnectors = 0.75
    ))
    dev.off()
    
    # MA plot
    png(sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\ma_plot_%s_vs_%s.png", comp_group_name, ref_group_name), width=800, height=600)
    plotMA(res, main=sprintf("MA Plot: %s vs %s", comp_group_name, ref_group_name))
    dev.off()
    
    # Save results
    write.csv(as.data.frame(res), sprintf("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\differential_expression_results_%s_vs_%s.csv", comp_group_name, ref_group_name))
    
    cat("Comparison completed successfully.\n\n")
}

# Perform group pair comparisons
group_pairs <- list(
    list(c("A", "B"), c("C", "D")),
    list(c("A", "C"), c("B", "D")),
    list(c("A", "D"), c("B", "C"))
)

for (pair in group_pairs) {
    perform_group_pair_comparison(dds, pair[[1]], pair[[2]])
}






###### Calculate volcano distances for all comparisons ######

### Assuming your DESeq2 results are stored in 'res'
# and res is a data frame (if it's not, convert it first with as.data.frame(res))

# Calculate -log10(p-value)
# Create a copy of res
res_volcano <- as.data.frame(res)

# Extract conditions from the comparison
conditions <- strsplit(res@elementMetadata$description[2], " vs ")[[1]]
comparison_condition <- sub(".*\\s", "", conditions[1])
reference_condition <- conditions[2]

# Add conditions as new columns
res_volcano$comparison_condition <- comparison_condition
res_volcano$reference_condition <- reference_condition

# Calculate -log10(p-value)
res_volcano$neg_log10_pvalue <- -log10(res_volcano$pvalue)

# Calculate distance from (0,0)
res_volcano$distance_from_center <- sqrt(res_volcano$log2FoldChange^2 + res_volcano$neg_log10_pvalue^2)





# Get list of CSV files in the folder
folder_path <- "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots"
csv_files <- list.files(folder_path, pattern = "differential_expression_results_.*\\.csv", full.names = TRUE)

# Remove files containing "all" from the list
csv_files <- csv_files[!grepl("all", csv_files)]

# Initialize an empty list to store dataframes
df_list <- list()

# Process each CSV file
for (file_name in csv_files) {
  # Load CSV file
  res <- read.csv(file_name)
  
  # Extract conditions from the file name
  conditions <- strsplit(sub("\\.csv$", "", basename(file_name)), "_")[[1]]
  comparison_condition <- conditions[length(conditions) - 2]
  reference_condition <- conditions[length(conditions)]
  
  # Add conditions as new columns
  res$comparison_condition <- comparison_condition
  res$reference_condition <- reference_condition
  
  # Calculate -log10(p-value)
  res$neg_log10_pvalue <- -log10(res$pvalue)
  
  # Calculate distance from (0,0)
  res$distance_from_center <- sqrt(res$log2FoldChange^2 + res$neg_log10_pvalue^2)
  
  # Add to list of dataframes
  df_list[[length(df_list) + 1]] <- res
}

# Merge all dataframes into a single master dataframe
master_df <- do.call(rbind, df_list)

# Print the dimensions of the master dataframe
print(paste("Master dataframe dimensions:", nrow(master_df), "rows,", ncol(master_df), "columns"))
#Save the master dataframe as a CSV file
write.csv(master_df, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\master_volcano_dataframe.csv", row.names = FALSE)


library(dplyr)
# Calculate z-scores for distance_from_center for each comparison
master_df <- master_df %>%
  group_by(comparison_condition, reference_condition) %>%
  mutate(
    mean_distance = mean(distance_from_center, na.rm = TRUE),
    sd_distance = sd(distance_from_center, na.rm = TRUE),
    z_score = (distance_from_center - mean_distance) / sd_distance
  ) %>%
  ungroup()

# Update the master dataframe CSV file with the new z-score column
write.csv(master_df, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\master_volcano_dataframe_with_zscores.csv", row.names = FALSE)



# Print summary of z-scores and count of rows more than 6 std away from mean
summary_stats <- master_df %>%
  group_by(comparison_condition, reference_condition) %>%
  summarise(
    min_z = min(z_score, na.rm = TRUE),
    max_z = max(z_score, na.rm = TRUE),
    mean_z = mean(z_score, na.rm = TRUE),
    median_z = median(z_score, na.rm = TRUE),
    count_6std_away = sum(abs(z_score) > 6, na.rm = TRUE)
  )

print(summary_stats)

print(summary_stats %>% select(comparison_condition, reference_condition, count_6std_away))


# Create a subset dataframe with rows where z_score is more than 6 std away
subset_df <- master_df %>%
  filter(abs(z_score) > 6)

# Print summary of the subset dataframe
print(paste("Number of rows in subset dataframe:", nrow(subset_df)))
print(head(subset_df))

# Save the subset dataframe as a CSV file
write.csv(subset_df, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\subset_dataframe_zscore_gt_6.csv", row.names = FALSE)



##### Fetch gene names from WormBase BioMart#####

###Tests
#BiocManager::install("biomaRt")
library(biomaRt)

# Connect to WormBase BioMart
wormbase <- useMart(biomart = "ENSEMBL_MART_ENSEMBL",
                    dataset = "celegans_gene_ensembl",
                    host = "www.ensembl.org")

# Define your gene IDs
gene_ids <- c("WBGene00021002", "WBGene00020543", "WBGene00002051", "WBGene00003595")

get_available_attributes <- function(mart) {
  attributes <- listAttributes(mart)
  return(attributes[grep("description", attributes$name, ignore.case = TRUE), ])
}
family_attributes <- get_available_attributes(wormbase)



# Fetch gene names
tryCatch({
  results <- getBM(attributes = c("wormbase_gene", "external_gene_name", "description", "goslim_goa_description", "entrezgene_description"),
                   filters = "wormbase_gene",
                   values = gene_ids,
                   mart = wormbase)
  
  print(results)
}, error = function(e) {
  cat("Error occurred:", conditionMessage(e), "\n")
  
  # Print available attributes if there's an error
  cat("\nAvailable attributes:\n")
  print(head(listAttributes(wormbase), 20))
})


### Add gene names to the subset dataframe

library(biomaRt)

# Connect to WormBase BioMart
wormbase <- useMart(biomart = "ENSEMBL_MART_ENSEMBL",
                    dataset = "celegans_gene_ensembl",
                    host = "www.ensembl.org")

# Get unique gene IDs from subset_df
gene_ids <- unique(subset_df$X)

# Fetch gene names
tryCatch({
  results <- getBM(attributes = c("wormbase_gene", "external_gene_name", "description"),
                   filters = "wormbase_gene",
                   values = gene_ids,
                   mart = wormbase)
  
  # Merge results with subset_df
  subset_df <- merge(subset_df, results, by.x = "X", by.y = "wormbase_gene", all.x = TRUE)
  
  # Rename the column for clarity
  names(subset_df)[names(subset_df) == "external_gene_name"] <- "gene_name"
  
  # Print the first few rows of the updated dataframe
  print(head(subset_df))
  
  # Save the updated subset dataframe as a CSV file
  write.csv(subset_df, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\subset_dataframe_zscore_gt_6_with_gene_names.csv", row.names = FALSE)
  
}, error = function(e) {
  cat("Error occurred:", conditionMessage(e), "\n")
  
  # Print available attributes if there's an error
  cat("\nAvailable attributes:\n")
  print(head(listAttributes(wormbase), 20))
})



# Load the updated subset dataframe
subset_df <- read.csv("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\subset_dataframe_zscore_gt_6_with_gene_names.csv")

# Count unique gene names with padj > 0.05
count_high_padj <- subset_df %>%
  filter(padj < 0.05) %>%
  distinct(gene_name) %>%
  nrow()

# Print the count
cat("Number of unique gene names with padj < 0.05:", count_high_padj, "\n") #100


# Count genes with padj < 0.05 that appear more than once
# Find repeated genes with padj < 0.05
repeated_genes <- subset_df %>%
  filter(padj < 0.05) %>%
  group_by(gene_name) %>%
  filter(n() > 1) %>%
  summarise(count = n(), comparisons = paste(unique(paste(comparison_condition, "vs", reference_condition)), collapse = ", ")) %>%
  arrange(desc(count))

# Print the repeated genes and their details
cat("Genes with padj < 0.05 that are repeated more than once:\n")
print(repeated_genes)

# Print the total count of repeated genes
cat("\nTotal number of genes with padj < 0.05 that are repeated more than once:", nrow(repeated_genes), "\n") #47




# Count genes with padj < 0.05 that appear more than once in single-character condition comparisons
# Find repeated genes with padj < 0.05
repeated_genes_single_char <- subset_df %>%
  filter(padj < 0.05,
         nchar(comparison_condition) == 1,
         nchar(reference_condition) == 1) %>%
  group_by(gene_name) %>%
  filter(n() > 1) %>%
  summarise(count = n(), comparisons = paste(unique(paste(comparison_condition, "vs", reference_condition)), collapse = ", ")) %>%
  arrange(desc(count))

# Print the repeated genes and their details for single-character condition comparisons
cat("Genes with padj < 0.05 that are repeated more than once in single-character condition comparisons:\n")
print(repeated_genes_single_char)

# Print the total count of repeated genes for single-character condition comparisons
cat("\nTotal number of genes with padj < 0.05 that are repeated more than once in single-character condition comparisons:", nrow(repeated_genes_single_char), "\n") #35




# Filter for single-character condition comparisons with padj < 0.05, keeping all rows
unique_genes_single_char <- subset_df %>%
  filter(nchar(comparison_condition) == 1,
         nchar(reference_condition) == 1,
         padj < 0.05)

# Get the count of unique gene names
unique_genes_count <- unique_genes_single_char %>%
  distinct(gene_name) %>%
  nrow()

# Print the count
cat("Number of unique gene names in single-character condition comparisons:", unique_genes_count, "\n") #73

# Print all rows that meet the filter conditions
cat("Genes in single-character condition comparisons with padj < 0.05:\n")
print(unique_genes_single_char)

# Save as csv
write.csv(unique_genes_single_char, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\unique_genes_single_char.csv", row.names = FALSE)



# Find unique genes in single-character condition comparisons that are not repeated and have padj < 1E-5
unique_significant_genes <- subset_df %>%
  filter(nchar(comparison_condition) == 1,
         nchar(reference_condition) == 1,
         padj < 0.05,
         padj < 1E-5) %>%
  anti_join(repeated_genes_single_char, by = "gene_name") %>%
  select(gene_name, padj, log2FoldChange, comparison_condition, reference_condition) %>%
  arrange(padj)

# Print the results
cat("Unique genes in single-character condition comparisons with padj < 1E-5 that are not repeated:\n")
print(unique_significant_genes %>%
  arrange(comparison_condition, reference_condition))

# Print the count of these genes
cat("\nNumber of unique genes with padj < 1E-5 that are not repeated:", nrow(unique_significant_genes), "\n") #14



##### Make a table with the log2FoldChange of the unique significant genes #####
# Load required libraries
library(tidyr)
library(dplyr)

# Assuming your dataframe is named 'unique_genes_single_char'
#Load the dataframe from the csv file
unique_genes_single_char <- read.csv("C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\unique_genes_single_char.csv")

# Step 1: Create a list of unique condition pairs in the desired order
condition_pairs <- c("B_A", "C_A", "B_C", "D_C", "D_B")

# Step 2: Create the first two rows of the new table
header_rows <- data.frame(
  gene_name = c("comparison_condition", "reference_condition"),
  matrix(c("B", "C", "B", "D", "D",
           "A", "A", "C", "C", "B"), 
         nrow = 2, ncol = length(condition_pairs), byrow = TRUE),
  pseudogene = c("pseudogene", "pseudogene"),
  description = c("description", "description"),
  padj = c("padj", "padj")
)
colnames(header_rows)[-1] <- c(condition_pairs, "pseudogene", "description", "padj")

# Step 3: Reshape the main data
reshaped_data <- unique_genes_single_char %>%
  select(gene_name, comparison_condition, reference_condition, log2FoldChange, pseudogene, description, padj) %>%
  mutate(pair = paste(comparison_condition, reference_condition, sep = "_")) %>%
  select(gene_name, pair, log2FoldChange, pseudogene, description, padj) %>%
  pivot_wider(
    names_from = pair,
    values_from = log2FoldChange,
    values_fill = NA  # Fill missing values with NA
  )

# Step 4: Convert all columns except gene_name to character
reshaped_data <- reshaped_data %>%
  mutate(across(-gene_name, as.character))

# Step 5: Group by gene_name and summarize
summarized_data <- reshaped_data %>%
  group_by(gene_name) %>%
  summarise(across(everything(), ~ if(all(is.na(.))) NA_character_ else na.omit(.)[1])) %>%
  ungroup()

# Step 6: Create a separate dataframe for sorting
sorting_data <- summarized_data %>%
  select(gene_name, padj) %>%
  mutate(padj = as.numeric(padj)) %>%
  arrange(padj)

# Step 7: Use the sorted gene_names to reorder the main dataframe
final_data <- sorting_data %>%
  select(gene_name) %>%
  left_join(summarized_data, by = "gene_name")

# Step 8: Reorder the columns
final_data <- final_data %>%
  select(gene_name, all_of(condition_pairs), pseudogene, description, padj)

# Step 9: Replace NA values with blank spaces
final_data <- final_data %>%
  mutate(across(everything(), ~replace_na(., "")))

# Step 10: Combine the header rows with the reordered data
final_table <- bind_rows(header_rows, final_data)

# View the result
print(final_table)

# Optional: Check how many genes have multiple values
genes_with_multiple_values <- unique_genes_single_char %>%
  group_by(gene_name) %>%
  summarise(combinations = n()) %>%
  filter(combinations > 1)

print(paste("Number of genes with multiple combinations:", nrow(genes_with_multiple_values)))

#save as csv
write.csv(final_table, "C:\\Users\\aurel\\Documents\\GitHub\\phd\\rnaseq\\plots\\unique_significant_genes_table.csv", row.names = FALSE)


