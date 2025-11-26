## MEL_TDDLSTM
Dieses Repo enthält den Quellcode für das TDDLSTM mit STFT-Features.

Um zu beginnen bitte das Repo herunterladen und in einem Pycharm Projekt öffnen, dann ein Venv erstellen und die notwendigen Bibliotheken herunterladen. WICHTIG: Bitte drauf achten, dass tensorflow==2.10 und numpy=1.24.0 verwendet wird.

Nach herunterladen der bibliotheken einfach "main.py" ausführen um zu trainieren und "Prediction_Mirrored_Phase.py" um mit dem trainierten Modell vorhersagen zu machen.

Hinweis: Das Spectral Folding wurde hierbei als Vorschritt in der Datensatzerstellung umgesetzt, demnach wird einfach im Testing ein weiterer Ordner übergeben, der die gespiegelten Signale enthält.
