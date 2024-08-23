#兩兩smiles比對相似度
install.packages("webchem")
install.packages("rcdk")
install.packages("fingerprint")
install.packages("rcdklibs")

library(webchem)
library(rcdk)
library(fingerprint)
library(rcdklibs)

smiles1 <- "C([C@H]([C@H]([C@@H]([C@H](C=O)O)O)O)O)O"
smiles2 <- "O=P(O)(O)OCC[N+](C)(C)C"

# 將 SMILES 轉換為分子對象
mol1 <- parse.smiles(smiles1)[[1]]
mol2 <- parse.smiles(smiles2)[[1]]

# 計算分子指紋
fp1 <- get.fingerprint(mol1, type = "circular")
fp2 <- get.fingerprint(mol2, type = "circular")

# 計算 Tanimoto 相似度
similarity <- distance(fp1, fp2, method = "tanimoto")

print(similarity)


#多個smiles與特定target smiles比對相似度
install.packages("webchem")
install.packages("rcdk")
install.packages("fingerprint")
install.packages("rcdklibs")

library(webchem)
library(rcdk)
library(fingerprint)
library(rcdklibs)
library(rJava)



smiles_list <- c( 'C[N+](C)(C)CCOP(=O)(O)O.[Cl-]',
                  'OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
                  'O=C(OC)COC1=CC=C(C=C1OC)C(C=2C(=O)C(O)=C(C(=O)C2O)C(C3=CC=C(OCC(=O)OC)C(OC)=C3)CC(=O)OC)CC(=O)OC',
                  'COC1=CC=C(C=C1)C1=CC(=O)C2=C(O)C(OC)=C(OC3OC(COC4OC(C)C(O)C(OC(C)=O)C4OC(C)=O)C(O)C(O)C3O)C=C2O1',
                  'CCCCCCCCCCCCCCCCCC(=O)OCC(O)COP(O)(=O)OCC(N)C(O)=O',
                  'O=C(OCC(OC(=O)CCCCCCCC=CCCCCCCCC)COC(=O)CCCCCCCCCCCCCCC)CCCCCCCC=CCCCCCCCCC',
                  'O=C1C(O)=C(C=2C=C(C(=C(O)C2C1=CNC3=CC(OC)=CC=C3OC)C=4C(O)=C5C(=CNC6=CC(OC)=CC=C6OC)C(=O)C(O)=C(C5=CC4C)C(C)C)C)C(C)C',
                  'O=C(O)CCCCCCCCCCC',
                  'C[C@@]12[C@@H](O2)CC/C(C)=C/C=C(C(C)(O)C)\\CC/C(C)=C/CC1',
                  'O=C1CC2C3(CC1C(=C)C3O)C4CC5C6(C)CN(CC)C4C25C(O)CC6',
                  'OC1CCC2(C)C(CCC3C2CCC4(C)C3CC5OC6(NCC(C)CC6)C(C)C54)C1',
                  'CCCCCCCCCCCCCCCCCCCCCCCCCC(O)C(CO)NC(=O)C(O)CCCCCCCCC=C/CCCCCC')


# 定義5個目標SMILES字符串
target_smiles <- c("C(C(F)(F)F)(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F",
                   "C(C(C(C(C(F)(F)Cl)(F)F)(F)F)(F)F)(C(C(C(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F",
                   "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F",
                   "C(C(C(C(F)(F)Cl)(F)F)(F)F)(C(C(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F",
                   "C(C(C(C(F)(F)F)(F)F)(F)F)(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F",
                   "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F")

# 將 SMILES 轉換為分子對象的函數
smiles_to_mol <- function(smiles) {
  parse.smiles(smiles)[[1]]
}

# 計算相似度的函數
calculate_similarity <- function(mol1, mol2_smiles) {
  mol2 <- parse.smiles(mol2_smiles)[[1]]
  fp1 <- get.fingerprint(mol1, type = "circular")
  fp2 <- get.fingerprint(mol2, type = "circular")
  distance(fp1, fp2, method = "tanimoto")
}

# 轉換目標SMILES為分子對象
target_mols <- lapply(target_smiles, smiles_to_mol)

# 創建相似度矩陣
similarity_matrix <- matrix(0, nrow = length(smiles_list), ncol = length(target_smiles))
colnames(similarity_matrix) <- target_smiles
rownames(similarity_matrix) <- smiles_list

# 計算相似度
for (i in 1:length(smiles_list)) {
  mol1 <- parse.smiles(smiles_list[i])[[1]]
  for (j in 1:length(target_smiles)) {
    similarity_matrix[i, j] <- calculate_similarity(mol1, target_smiles[j])
  }
}

print(similarity_matrix)

# 將結果彙整為數據框格式
results <- data.frame(SMILES = rownames(similarity_matrix), similarity_matrix)


# 列印彙整結果並找出最佳匹配
for (i in 1:nrow(results)) {
  cat("SMILES:", results$SMILES[i], "\n")
  
  # 找到相似度最高的目標SMILES
  max_similarity <- max(as.numeric(results[i, -1]))
  best_match_index <- which(as.numeric(results[i, -1]) == max_similarity)
  best_match_smiles <- colnames(results)[best_match_index + 1] # +1 因为第一列是SMILES
  
  # 确保最佳匹配SMILES没有被转义处理
  best_match_smiles <- gsub("\n", "", best_match_smiles) # 移除换行符（如果有）
  
  # 列印相似度和最佳匹配
  cat("Similarities:", paste(results[i, -1], collapse = ", "), "\n")
  cat("Best Match:", best_match_smiles, "with Similarity:", max_similarity, "\n\n")
}








