package tfgAlejandroNavarroMesa;
import javafx.geometry.Orientation;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import javafx.scene.control.Button;
import javafx.scene.control.Tooltip;
import javafx.scene.control.Separator;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.BorderPane;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.common.ColorTools;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.analysis.images.ContourTracing;
import javafx.application.Platform;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.nio.file.Files;
import java.util.*;
import java.util.List;
import java.util.prefs.Preferences;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.dialogs.Dialogs;

public class ExtensionTFG implements QuPathExtension{

    private static final Logger logger = LoggerFactory.getLogger(ExtensionTFG.class);
    public static class PatchInfo {
        int x10,y10, predicted_label;
        double [] probabilities, adjusted_probabilities;
        double weight;
    }

    private static final double ICON_BTN_SIZE = 32.0;
    private final Button modelButton = createIconButton("\uD83E\uDD16", "Selecciona el modelo de la predicción");
    private final Button strategyButton = createIconButton("⧉", "Selecciona la estrategia de solapamiento");
    private final Button clasificationButton =createIconButton("\uD83D\uDCCA", "Selecciona el tipo de clasificación");
    private final Button visibilityButton = createIconButton("⛔", "Visibiliza u oculta las predicciones");
    private final Button settingsButton = createIconButton("\uD83D\uDCC1", "Configura la ruta de los archivos CSV");

    private final Label statusModel = new Label("Modelo: ");
    private final Label statusClassification = new Label("Tipo de Clasificación: ");
    private final Label statusStrategy = new Label("Estrategia de solapamiento: ");
    private final Label statusVisibility = new Label("Visibilidad: ");

    private List<PathObject> savedObjects = new ArrayList<>();
    private boolean show = true;
    private boolean predictions = false;

    private static Button createIconButton(String icon, String tooltip){
        Button button = new Button(icon);
        button.setTooltip(new Tooltip(tooltip));
        button.setMinSize(ICON_BTN_SIZE, ICON_BTN_SIZE);
        button.setMaxSize(ICON_BTN_SIZE, ICON_BTN_SIZE);
        button.setPrefSize(ICON_BTN_SIZE, ICON_BTN_SIZE);
        button.setStyle("-fx-font-size: 16px; -fx-padding: 0;");
        return button;
    }

    @Override
    public void installExtension(QuPathGUI qupath){
        Preferences prefs = Preferences.userRoot().node("TFG");

        settingsButton.setOnAction(event -> {
            event.consume();
            String savedPath = prefs.get("csvdir", "C:\\Users\\Alex\\Downloads\\UNIVERSIDAD2526\\TFG");
            String baseCsvDir = Dialogs.showInputDialog("Ruta de los archivos CSV", "Introduce la carpeta donde están alojados los archivos CSV:", savedPath);
            if (baseCsvDir != null && !baseCsvDir.trim().isEmpty()) {
                prefs.put("csvdir", baseCsvDir);
                Dialogs.showPlainMessage("Ruta guardada", "La ruta ha sido guardada con éxito");
            }
        });

        modelButton.setOnAction(event->{
            event.consume();
            ImageServer<BufferedImage> server = qupath.getViewer().getServer();
            if (server == null) {
                Dialogs.showErrorMessage("Error", "Tiene que abrir una imagen");
                return;
            }
            String wsiName = server.getMetadata().getName().replace(".tiff", "");

            List<String> models = Arrays.asList(
                    "CONCH_SVM",
                    "CONCH_Red_Propia",
                    "CONCH_Random_Forest"
            );

            String savedModelPref = prefs.get("model_" + wsiName, "");
            String defaultChoice = savedModelPref.isEmpty() ? null : savedModelPref;
            String selectedModel = Dialogs.showChoiceDialog(
                    "Selección de modelo",
                    "Selecciona el modelo para realizar las predicciones:",
                    models,
                    defaultChoice
            );

            if (selectedModel != null && !selectedModel.equals(savedModelPref)){
                prefs.put("model_" + wsiName, selectedModel);
                if (prefs.get("clasification_" + wsiName, "").isEmpty()){
                    prefs.put("clasification_" + wsiName, "Multiclase");
                }
                if (prefs.get("strategy_" + wsiName, "").isEmpty()){
                    prefs.put("strategy_" + wsiName, "Voto ponderado por Entropía con Penalización Asimétrica");
                }

                String clasification = prefs.get("clasification_" + wsiName, "");
                String csvDir = prefs.get("csvdir","");
                String strategy = prefs.get("strategy_" + wsiName, "");
                clearObjects(qupath);
                updateButtons(qupath);
                new Thread(() -> drawPatches(qupath, strategy, csvDir, clasification, selectedModel)).start();
            }
        });

        strategyButton.setOnAction(event->{
            event.consume();
            ImageServer<BufferedImage> server = qupath.getViewer().getServer();
            if (server == null) {
                Dialogs.showErrorMessage("Error", "Tiene que abrir una imagen");
                return;
            }
            String wsiName = server.getMetadata().getName().replace(".tiff", "");

            List<String> overlapStrategies = Arrays.asList(
                    "Asignación del Último parche",
                    "Interpolación de Color",
                    "Voto por Mayoría Absoluta",
                    "Voto ponderado por Entropía con Penalización Asimétrica"
            );

            String savedStrategyPref = prefs.get("strategy_" + wsiName, "");
            String defaultChoice = savedStrategyPref.isEmpty() ? overlapStrategies.getFirst() : savedStrategyPref;
            String selectedStrategy = Dialogs.showChoiceDialog(
                    "Estrategia de solapamiento",
                    "Selecciona la estrategia para resolver los solapamientos:",
                    overlapStrategies,
                    defaultChoice
            );

            if (selectedStrategy != null && !selectedStrategy.equals(savedStrategyPref)){
                prefs.put("strategy_" + wsiName, selectedStrategy);
                if (prefs.get("clasification_" + wsiName, "").isEmpty()){
                    prefs.put("clasification_" + wsiName, "Multiclase");
                }
                if (prefs.get("model_" + wsiName, "").isEmpty()){
                    prefs.put("model_" + wsiName, "CONCH_SVM");
                }

                String clasification = prefs.get("clasification_" + wsiName, "");
                String csvDir = prefs.get("csvdir","");
                String model = prefs.get("model_" + wsiName, "");
                clearObjects(qupath);
                updateButtons(qupath);
                new Thread(() -> drawPatches(qupath, selectedStrategy, csvDir, clasification, model)).start();
            }
        });

        clasificationButton.setOnAction(event->{
            event.consume();
            ImageServer<BufferedImage> server = qupath.getViewer().getServer();
            if (server == null) {
                Dialogs.showErrorMessage("Error", "Tiene que abrir una imagen");
                return;
            }
            String wsiName = server.getMetadata().getName().replace(".tiff", "");

            List<String> types = Arrays.asList("Binaria", "Multiclase");

            String savedtypePref = prefs.get("clasification_" + wsiName, "");
            String defaultChoice = savedtypePref.isEmpty() ? types.getFirst() : savedtypePref;
            String selectedtype = Dialogs.showChoiceDialog(
                    "Tipo de clasificación",
                    "Selecciona el tipo de clasificación a utilizar:",
                    types,
                    defaultChoice);

            if (selectedtype != null && !selectedtype.equals(savedtypePref)){
                prefs.put("clasification_" + wsiName, selectedtype);
                if (prefs.get("model_" + wsiName, "").isEmpty()){
                    prefs.put("model_" + wsiName, "CONCH_SVM");
                }
                if (prefs.get("strategy_" + wsiName, "").isEmpty()){
                    prefs.put("strategy_" + wsiName, "Voto ponderado por Entropía con Penalización Asimétrica");
                }

                String csvDir = prefs.get("csvdir", "");
                String strategy = prefs.get("strategy_" + wsiName, "");
                String model = prefs.get("model_" + wsiName, "");
                clearObjects(qupath);
                updateButtons(qupath);
                new Thread(() -> drawPatches(qupath, strategy, csvDir, selectedtype, model)).start();
            }
        });

        visibilityButton.setOnAction(event->{
            event.consume();
            if (qupath.getViewer().getServer() == null) {
                Dialogs.showErrorMessage("Error", "Tiene que abrir una imagen");
                return;
            }
            if (!predictions){
                return;
            }
            var hierarchy = qupath.getViewer().getHierarchy();
            if (hierarchy == null){
                return;
            }

            if (show){
                savedObjects = new ArrayList<>(hierarchy.getAllObjects(false));
                hierarchy.removeObjects(savedObjects, true);
                show = false;
            } else{
                hierarchy.addObjects(savedObjects);
                hierarchy.fireHierarchyChangedEvent(qupath);
                savedObjects.clear();
                show = true;
            }
            updateButtons(qupath);
        });

        qupath.getViewer().imageDataProperty().addListener((ignoredObs, ignoredOldData, newData) -> {
            if (newData != null){
                Platform.runLater(()-> {
                    savedObjects.clear();
                    show = true;

                    ImageServer<BufferedImage> server = qupath.getViewer().getServer();
                    if (server != null){
                        String wsiName = server.getMetadata().getName().replace(".tiff", "");
                        String savedModel = prefs.get("model_" + wsiName, "");

                        if (!savedModel.isEmpty()) {
                            var hierarchy = qupath.getViewer().getHierarchy();
                            boolean hasAnnotations = hierarchy != null && !hierarchy.getAnnotationObjects().isEmpty();

                            if (hasAnnotations) {
                                predictions = true;
                            } else {
                                prefs.remove("model_" + wsiName);
                                prefs.remove("clasification_" + wsiName);
                                prefs.remove("strategy_" + wsiName);
                                predictions = false;
                            }
                        } else {
                            predictions = false;
                        }
                    } else {
                        predictions = false;
                    }
                    updateButtons(qupath);
                });
            }
        });

        Platform.runLater(()->{
            var items = qupath.getToolBar().getItems();
            items.addFirst(new Separator());
            items.addFirst(visibilityButton);
            items.addFirst(strategyButton);
            items.addFirst(clasificationButton);
            items.addFirst(modelButton);
            items.addFirst(settingsButton);

            HBox statusBar = buildStatusBar();
            javafx.scene.Parent root = qupath.getStage().getScene().getRoot();
            BorderPane mainBorderPlane = findMainBorderPlane(root);
            if (mainBorderPlane != null){
                javafx.scene.Node oldBottom = mainBorderPlane.getBottom();
                if (oldBottom != null){
                    javafx.scene.layout.VBox newBottom = new javafx.scene.layout.VBox(statusBar, oldBottom);
                    mainBorderPlane.setBottom(newBottom);
                } else {
                    mainBorderPlane.setBottom(statusBar);
                }
            }
            updateButtons(qupath);
        });
    }

    private BorderPane findMainBorderPlane(javafx.scene.Node node){
        if (node instanceof  BorderPane){
            return (BorderPane) node;
        }
        if (node instanceof javafx.scene.Parent){
            for (javafx.scene.Node child : ((javafx.scene.Parent) node).getChildrenUnmodifiable()){
                BorderPane result = findMainBorderPlane(child);
                if (result != null){
                    return  result;
                }
            }
        }
        return null;
    }

    private HBox buildStatusBar(){
        statusModel.setStyle("-fx-font-size: 11px;");
        statusClassification.setStyle("-fx-font-size: 11px;");
        statusStrategy.setStyle("-fx-font-size: 11px;");
        statusVisibility.setStyle("-fx-font-size: 11px;");

        Separator separator1 = new Separator(Orientation.VERTICAL);
        Separator separator2 = new Separator(Orientation.VERTICAL);
        Separator separator3 = new Separator(Orientation.VERTICAL);

        HBox statusBar = new HBox(12,
                statusModel, separator1, statusClassification, separator2, statusStrategy, separator3, statusVisibility
        );
        statusBar.setAlignment(Pos.CENTER_LEFT);
        statusBar.setPadding(new Insets(4,10,4,10));
        statusBar.setStyle("-fx-background-color: -fx-base;" + "-fx-border-color: -fx-box-border;" + "-fx-border-width: 1 0 0 0;");
        return  statusBar;
    }

    private void drawPatches(QuPathGUI qupath, String selectedStrategy, String baseCsvDir, String clasification, String model){
        if (model == null || model.isEmpty()){
            return;
        }
        try{
            ImageServer<BufferedImage> server = qupath.getViewer().getServer();
            if (server == null){
                Platform.runLater(() -> Dialogs.showErrorMessage("Error", "Tiene que abrir una imagen"));
                return;
            }

            String wsiName = server.getMetadata().getName().replace(".tiff", "");
            File csvFile = new File(baseCsvDir, wsiName + "_labels_and_coords_and_predictions_"+ model + ".csv");
            if (!csvFile.exists()){
                Platform.runLater(()-> Dialogs.showErrorMessage("Error", "CSV no encontrado: \n" + csvFile.getAbsolutePath() + "\n\nNombre WSI: " + wsiName));
                return;
            }
            int scalerFactor = 4;
            int patchSize10 = 512;
            int patchSize40 = patchSize10 * scalerFactor;
            int stride10 = (int)(patchSize10 * (1 -0.5));
            int stride40 = stride10 * scalerFactor;
            double downsample = 4.0;
            int threshold = 200;
            int opacity = 120;
            double epsilon_weight = 1e-5;
            double epsilon_prob = 1e-9;
            double[] severityExponents = {1.0,1.0, 0.85, 0.7};
            boolean binary = clasification.equals("Binaria");

            PathClass.getInstance("Tejido sano", ColorTools.packARGB(opacity, 150, 255, 100));
            PathClass classNC = PathClass.getInstance("Tejido sano", ColorTools.packARGB(opacity, 150, 255, 100));
            PathClass classG3 = PathClass.getInstance("Gleason 3", ColorTools.packARGB(opacity, 255, 255, 0));
            PathClass classG4 = PathClass.getInstance("Gleason 4", ColorTools.packARGB(opacity, 255, 120, 0));
            PathClass classG5 = PathClass.getInstance("Gleason 5", ColorTools.packARGB(opacity, 255, 0, 0));
            PathClass classCancer = PathClass.getInstance("Tejido Canceroso", ColorTools.packARGB(opacity, 255, 0, 0));

            Map<Integer, PathClass> classMap = new HashMap<>();
            classMap.put(0, classNC);
            if (binary){
                classMap.put(1, classCancer);
            }else{
                classMap.put(1, classG3);
                classMap.put(2, classG4);
                classMap.put(3, classG5);
            }

            List<String> lines = Files.readAllLines(csvFile.toPath());
            String[] header = lines.getFirst().split(",");
            Map<String, Integer> colMap = new HashMap<>();
            for(int i = 0; i < header.length; i++){
                colMap.put(header[i].trim(), i);
            }
            List<PatchInfo> patches = new ArrayList<>();
            for (int i=1; i<lines.size(); i++){
                String line = lines.get(i).trim();
                if(line.isEmpty()){
                    continue;
                }
                String[] cols = line.split(",");

                PatchInfo patch = new PatchInfo();
                patch.x10 = (int) Double.parseDouble(cols[colMap.get("x_ini")]);
                patch.y10 = (int) Double.parseDouble(cols[colMap.get("y_ini")]);
                int rawLabel = Integer.parseInt(cols[colMap.get("predicted_label")]);

                if (binary){
                    patch.predicted_label = (rawLabel > 0) ? 1 : 0;
                    double pCancer = epsilon_prob;
                    double pNC = epsilon_prob;
                    for (int p =0; p<4; p++) {
                        String name = p + "_probability";
                        if (colMap.containsKey(name)) {
                            double prob = Double.parseDouble(cols[colMap.get(name)]);
                            if (p == 0) {
                                pNC = Math.max(prob, epsilon_prob);
                            } else {
                                pCancer += prob;
                            }
                        }
                    }
                    pCancer = Math.max(pCancer, epsilon_prob);
                    double total = pNC + pCancer;
                    patch.probabilities = new double[]{pNC / total, pCancer / total};
                } else {
                    patch.predicted_label = rawLabel;
                    patch.probabilities = new double[4];
                    for (int p = 0; p < 4; p++) {
                        String name = p + "_probability";
                        if (colMap.containsKey(name)) {
                            patch.probabilities[p] = Math.max(Double.parseDouble(cols[colMap.get(name)]), epsilon_prob);
                        } else {
                            patch.probabilities[p] = (p == rawLabel) ? 1.0 : epsilon_prob;
                        }
                    }
                }
                patches.add(patch);
            }
            int numClasses = binary ? 2 : 4;
            double[] exponents = binary ? new double[]{1.0, 1.0} : severityExponents;
            for (PatchInfo patch : patches){
                patch.weight = 1.0 / (entropy(patch.probabilities) + epsilon_weight);
                double[] adjusted = new double[4];

                for (int i=0; i < patch.probabilities.length; i++){
                    adjusted[i] = Math.pow(patch.probabilities[i], exponents[i]);
                }
                patch.adjusted_probabilities = adjusted;
            }
            Map<String, PatchInfo> findPatch = new HashMap<>();
            for (PatchInfo p: patches){
                findPatch.put(p.x10 + "_" + p.y10, p);
            }
            int [][] offsets = {{0,0}, {-stride40,0}, {0, -stride40}, {-stride40, -stride40}};
            int minX = Integer.MAX_VALUE;
            int minY= Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE;
            int maxY = Integer.MIN_VALUE;
            for (PatchInfo cell : patches){
                int x40 = cell.x10;
                int y40 = cell.y10;
                if(x40 < minX){
                    minX = x40;
                }
                if(y40 < minY){
                    minY = y40;
                }
                if (x40 + patchSize40 > maxX){
                    maxX = x40 + patchSize40;
                }
                if (y40 + patchSize40 > maxY){
                    maxY = y40 + patchSize40;
                }
            }
            int maskDownscale = (int) downsample;
            int maskW = (maxX -minX) / maskDownscale;
            int maskH = (maxY - minY) / maskDownscale;
            int [] labelMask = new int[maskW * maskH];
            ImagePlane plane = ImagePlane.getDefaultPlane();
            final int finalMinX = minX;
            final int finalMinY = minY;
            Map<String, Integer> winnerMap = new java.util.concurrent.ConcurrentHashMap<>();
            patches.parallelStream().forEach (cell-> {
                List<PatchInfo> neighbors = new ArrayList<>();
                for (int[] offset : offsets) {
                    PatchInfo neighbor = findPatch.get((cell.x10 + offset[0]) + "_" + (cell.y10 + offset[1]));
                    if (neighbor != null) {
                        neighbors.add(neighbor);
                    }
                }
                if (neighbors.isEmpty()) {
                    return;
                }

                int winner = 0;
                PathClass[] blendedHolder = {null};
                switch (selectedStrategy) {
                    case "Voto ponderado por Entropía con Penalización Asimétrica":
                        double[] scores = new double[numClasses];
                        for (PatchInfo neighbor : neighbors) {
                            for (int s = 0; s < numClasses; s++) {
                                scores[s] += neighbor.weight * neighbor.adjusted_probabilities[s];
                            }
                        }
                        for (int s = 1; s < numClasses; s++) {
                            if (scores[s] > scores[winner]) {
                                winner = s;
                            }
                        }
                        break;
                    case "Voto por Mayoría Absoluta":
                        int[] votes = new int[numClasses];
                        for (PatchInfo neighbor : neighbors) {
                            votes[neighbor.predicted_label]++;
                        }
                        for (int s = 1; s < numClasses; s++) {
                            if (votes[s] > votes[winner]) {
                                winner = s;
                            }
                        }
                        break;
                    case "Interpolación de Color":
                        if (binary) {
                            winner = neighbors.stream().anyMatch(n -> n.predicted_label > 0) ? 1 : 0;
                        } else {
                            List<PatchInfo> colorNeighbors = neighbors.stream().filter(n -> n.predicted_label > 0).toList();
                            if (colorNeighbors.isEmpty()) {
                                break;
                            }
                            int[][] rgb = {{0, 0, 0}, {255, 255, 0}, {255, 120, 0}, {255, 0, 0}};
                            int totalR = 0;
                            int totalG = 0;
                            int totalB = 0;
                            for (PatchInfo n : colorNeighbors) {
                                totalR += rgb[n.predicted_label][0];
                                totalG += rgb[n.predicted_label][1];
                                totalB += rgb[n.predicted_label][2];
                            }
                            int finalR = totalR / colorNeighbors.size();
                            int finalG = totalG / colorNeighbors.size();
                            int finalB = totalB / colorNeighbors.size();
                            blendedHolder[0] = PathClass.getInstance("Interpolacion_" + finalR + "_" + finalG + "_" + finalB, ColorTools.packARGB(opacity, finalR, finalG, finalB));
                            winner = -1;
                        }
                        break;
                    case "Asignación del Último parche":
                        for (PatchInfo neighbor : neighbors) {
                            if (neighbor.predicted_label >= winner) {
                                winner = neighbor.predicted_label;
                            }
                        }
                        break;
                    default:
                        winner = cell.predicted_label;
                        break;
                }
                int labelToWrite;
                if (winner == -1 && blendedHolder[0] != null) {
                    int packedColor = blendedHolder[0].getColor();
                    int r = ColorTools.red(packedColor);
                    int g = ColorTools.green(packedColor);
                    int b = ColorTools.blue(packedColor);
                    labelToWrite = 1000 + r * 65536 + g * 256 + b;
                } else {
                    labelToWrite = winner;
                }
                winnerMap.put(cell.x10 + "_" + cell.y10, labelToWrite);
            });

            patches.parallelStream().forEach(cell ->{
                Integer labelToWrite = winnerMap.get(cell.x10 + "_" + cell.y10);
                if (labelToWrite == null) {
                    return;
                }

                int px0 = (cell.x10 - finalMinX) / maskDownscale;
                int py0 = (cell.y10 - finalMinY) / maskDownscale;
                int pxEnd = Math.min(px0 + patchSize40 / maskDownscale, maskW);
                int pyEnd = Math.min(py0 + patchSize40 / maskDownscale, maskH);
                try{
                    RegionRequest patchRequest = RegionRequest.createInstance(
                            server.getPath(), maskDownscale, cell.x10, cell.y10, patchSize40, patchSize40, plane
                    );
                    BufferedImage patchImg = server.readRegion(patchRequest);
                    if (patchImg != null){
                        int pw = patchImg.getWidth();
                        int ph = patchImg.getHeight();
                        BufferedImage grayImg = new BufferedImage(pw,ph,BufferedImage.TYPE_BYTE_GRAY);
                        grayImg.createGraphics().drawImage(patchImg, 0, 0,null);
                        Raster raster = grayImg.getRaster();
                        synchronized (labelMask){
                            for (int py = py0; py < pyEnd; py++){
                                for (int px = px0; px < pxEnd; px++){
                                    int localX = px -px0;
                                    int localY = py - py0;
                                    if (localX >= pw || localY >= ph){
                                        continue;
                                    }
                                    float gray = raster.getSample(localX, localY, 0);
                                    if (gray < threshold){
                                        int idx = py * maskW + px;
                                        labelMask[idx] = labelToWrite + 1;
                                    }
                                }
                            }
                        }
                    } else{
                        logger.warn("No se pudo leer la imagen del parche en ({}, {}); se omite este parche", cell.x10, cell.y10);
                    }
                } catch (Exception imgEx){
                    logger.warn("Error al leer la imagen del parche en ({}, {}); se omite este parche", cell.x10, cell.y10, imgEx);
                }
            });

            Set<Integer> uniqueLabels = new HashSet<>();
            for(int v: labelMask){
                if (v>0){
                    uniqueLabels.add(v);
                }
            }
            List<PathObject> tiles = new ArrayList<>();
            for (int labelPlusOne : uniqueLabels){
                float[] binaryPixels = new float[maskW * maskH];
                for (int i =0; i<labelMask.length; i++){
                    binaryPixels[i] = (labelMask[i] == labelPlusOne) ? 255f : 0f;
                }
                var simpleImage = SimpleImages.createFloatImage(binaryPixels, maskW, maskH);
                RegionRequest maskRequest = RegionRequest.createInstance(
                        qupath.getViewer().getServer().getPath(),
                        maskDownscale,
                        minX,
                        minY,
                        maskW * maskDownscale,
                        maskH * maskDownscale,
                        plane
                );
                ROI mergedRoi = ContourTracing.createTracedROI(simpleImage, 127.5, Double.MAX_VALUE, maskRequest);
                if (mergedRoi == null || mergedRoi.isEmpty()){
                    continue;
                }
                int rawLabel = labelPlusOne -1;
                PathClass pClass;
                if (rawLabel >= 1000){
                    int rgb = rawLabel - 1000;
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    pClass = PathClass.getInstance("Interpolacion_" + r + "_" + g + "_" + b, ColorTools.packARGB(opacity, r, g, b));
                } else {
                    pClass = classMap.getOrDefault(rawLabel, classNC);
                }
                PathObject obj = PathObjects.createAnnotationObject(mergedRoi, pClass);
                obj.setLocked(true);
                tiles.add(obj);
            }
            int numParchesPintados = patches.size();
            PathClass[] legendClasses;
            String[] legendNames;
            if (binary){
                legendClasses = new PathClass[]{classNC, classCancer};
                legendNames = new String[]{"Tejido Sano", "Tejido Canceroso"};
            }else{
                legendClasses = new PathClass[]{classNC, classG3, classG4, classG5};
                legendNames = new String[]{"Tejido Sano (NC)", "Gleason 3", "Gleason 4", "Gleason 5"};
            }

            int maxDim = Math.max(server.getWidth(), server.getHeight());

            int legendPatchSize = Math.max(80, maxDim / 45);

            int legendMarginLeft = legendPatchSize * 5;

            int legendW = legendMarginLeft + (legendPatchSize * 3);
            int legendH = (legendClasses.length * (legendPatchSize + legendPatchSize)) + legendPatchSize;

            int[] legendOrigin = putLegend(server, patches, legendW, legendH, patchSize40);

            for (int i=0; i<legendClasses.length; i++){
                int x = legendOrigin[0] + legendMarginLeft;
                int y = legendOrigin[1] + legendPatchSize + i * (legendPatchSize + legendPatchSize);
                ROI roi = ROIs.createRectangleROI(x,y,legendPatchSize, legendPatchSize, plane);
                PathObject label = PathObjects.createAnnotationObject(roi, legendClasses[i]);
                label.setName(legendNames[i]);
                label.setLocked(true);
                tiles.add(label);
            }

            final float dynamicFontSize = (float) Math.max(30.0, legendPatchSize * 0.04);

            Platform.runLater(()->{
                clearObjects(qupath);
                try{
                    qupath.getViewer().getOverlayOptions().fontSizeProperty().set(dynamicFontSize);
                } catch (Exception ignored){}
                qupath.getViewer().getHierarchy().addObjects(tiles);

                Preferences prefsLocal = Preferences.userRoot().node("TFG");
                show = Boolean.parseBoolean(prefsLocal.get("visibility_" + wsiName, "true"));
                predictions = true;

                if (!show){
                    savedObjects = new ArrayList<>(qupath.getViewer().getHierarchy().getAllObjects(false));
                    qupath.getViewer().getHierarchy().removeObjects(savedObjects, true);
                } else {
                    savedObjects.clear();
                }

                updateButtons(qupath);
                Dialogs.showPlainMessage("Predicción terminada", "Se han añadido " +numParchesPintados + " parches predichos en " + wsiName + " con la estrategia " + selectedStrategy + " y la clasificación " + clasification);
            });

        }catch (Exception ex){
            logger.error("Ha ocurrido un error durante la predicción y dibujado de parches", ex);
            Platform.runLater(()->Dialogs.showErrorMessage("Error", "Ha sucedido el siguiente fallo: " + ex.getMessage()));
        }
    }

    private  static double entropy(double[] probs){
        double entropy = 0.0;
        for (double p : probs){
            entropy -= p * (Math.log(p) / Math.log(2.0));
        }
        return entropy;
    }

    private int[] putLegend(ImageServer<BufferedImage> server, List<PatchInfo> patches, int legendW, int legendH, int patchSize40){
        int fullH = server.getHeight();
        int fullW = server.getWidth();
        if (fullW <= 0 || fullH<=0){
            return new int[]{0,0};
        }

        int gridCols = 150;
        int gridRows = 150;
        double cellH =(double) fullH / gridRows;
        double cellW = (double) fullW / gridCols;
        boolean[][] occupied = new boolean[gridRows][gridCols];

        for (PatchInfo p : patches) {
            int cStart = Math.max(0, (int) (p.x10 / cellW));
            int rStart = Math.max(0, (int) (p.y10 / cellH));
            int cEnd = Math.min(gridCols - 1, (int) ((p.x10 + patchSize40) / cellW));
            int rEnd = Math.min(gridRows - 1, (int) ((p.y10 + patchSize40) / cellH));

            for (int r = rStart; r <= rEnd; r++) {
                for (int c = cStart; c <= cEnd; c++) {
                    occupied[r][c] = true;
                }
            }
        }

        int reqCols = Math.max(1,(int)Math.ceil(legendW / cellW));
        int reqRows = Math.max(1, (int)Math.ceil(legendH / cellH));
        int bestR = 0;
        int bestC = 0;
        long minPenalty = Long.MAX_VALUE;
        for (int r = 0; r<= gridRows - reqRows; r++){
            for (int c=0; c<=gridCols -reqCols; c++){
                int overlapc = 0;
                for (int cr = 0; cr<reqRows; cr++){
                    for(int cc = 0; cc<reqCols; cc++){
                        if (occupied[r + cr][c + cc]){
                            overlapc++;
                        }
                    }
                }

                long penalty = (long) overlapc * 10000000000L + ((long) r * r + (long) c * c);
                if (penalty < minPenalty){
                    minPenalty = penalty;
                    bestC = c;
                    bestR = r;
                }
            }
        }

        int finalX = (int)(bestC * cellW);
        int finalY = (int)(bestR * cellH);
        int padX = fullW / 20;
        int padY = fullH / 20;
        finalX = Math.clamp(finalX, padX, fullW - legendW - padX);
        finalY = Math.clamp(finalY, padY, fullH - legendH - padY);
        return new int[]{finalX, finalY};
    }

    private void clearObjects(QuPathGUI qupath){
        savedObjects.clear();
        predictions = false;

        Preferences prefs = Preferences.userRoot().node("TFG");
        ImageServer<BufferedImage> server = qupath.getViewer().getServer();
        if (server != null) {
            String wsiName = server.getMetadata().getName().replace(".tiff", "");
            show = Boolean.parseBoolean(prefs.get("visibility_" + wsiName, "true"));
        } else {
            show = true;
        }

        var hierarchy = qupath.getViewer().getHierarchy();
        if (hierarchy == null){
            return;
        }
        var remove = new ArrayList<>(hierarchy.getAllObjects(false));
        if (!remove.isEmpty()){
            hierarchy.removeObjects(remove, true);
        }
    }

    private void updateButtons(QuPathGUI qupath){
        Preferences prefs = Preferences.userRoot().node("TFG");
        String model = "";
        String classification = "";
        String strategy = "";

        if (qupath.getViewer() != null && qupath.getViewer().getServer() != null) {
            String wsiName = qupath.getViewer().getServer().getMetadata().getName().replace(".tiff", "");
            model = prefs.get("model_" + wsiName, "");
            classification = prefs.get("clasification_" + wsiName, "");
            strategy = prefs.get("strategy_" + wsiName, "");
        }

        String displayStrategy = strategy;
        if (strategy.contains("Voto ponderado por Entropía")){
            if ("Binaria".equals(classification)){
                displayStrategy = "Voto ponderado por Entropía";
            } else{
                displayStrategy = "Voto ponderado por Entropía con Penalización Asimétrica";
            }
        }

        modelButton.setTooltip(new Tooltip(
                "Modelo: " + (model.isEmpty() ? "no seleccionado" : model.replace('_', ' '))));
        clasificationButton.setTooltip(new Tooltip(
                "Clasificación: " + (classification.isEmpty() ? "no seleccionada" : classification)));
        strategyButton.setTooltip(new Tooltip(
                "Estrategia: " + (strategy.isEmpty() ? "no seleccionada" : strategy)));

        visibilityButton.setTooltip(new Tooltip(
                !predictions ? "No hay predicciones para mostrar u ocultar" : (show ? "Ocultar predicciones" : "Mostrar predicciones")));
        visibilityButton.setText(show ? "⛔" : "\uD83D\uDC41");

        final String finalModel = model;
        final String finalClassification = classification;
        final String finalDisplayStrategy = displayStrategy;

        Platform.runLater(()->{
            statusModel.setText("\uD83E\uDD16 Modelo: " + (finalModel.isEmpty()   ? "—" : finalModel.replace('_', ' ')));
            statusClassification.setText("\uD83D\uDCCA Clasificación: " + (finalClassification.isEmpty() ? "—" : finalClassification));
            statusStrategy.setText("⧉ Estrategia: " + (finalDisplayStrategy.isEmpty() ? "—" : finalDisplayStrategy));
            if (!predictions){
                statusVisibility.setText("\uD83D\uDC41 Visibilidad: —");
            } else{
                statusVisibility.setText(show ? "\uD83D\uDC41 Visible" : "\uD83D\uDEAB Oculto");
            }
        });
    }
    @Override
    public String getName(){
        return "Extensión";
    }

    @Override
    public String getDescription(){
        return "Herramienta para gestionar el solapamiento de parches";
    }
}