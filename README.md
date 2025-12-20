2025にpublishされたHuangの論文の再現実装を行なっています。

基本的にはおいてあるコードを回すだけですが、vlasov_simulationのコードはtraining_dataとtest_dataを作成するために2回回す必要があります。

順番としては、

vlasov_simulation.pyを dt=5e-3, savedirをtrainingとして実行

vlasov_simulation.pyを dt=2e-3, savedirをtestとして実行

machine_learning.pyの実行

fluid_hpclosure.pyの実行

fluid_mlclosure.pyの実行

plot.pyの実行

という手順で再現実装が完了します。

machine learningはcpu環境でも動作することが確認できました。