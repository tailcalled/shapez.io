import { generateMatrixRotations } from "../../core/utils";
import { enumDirection, Vector } from "../../core/vector";
import { ItemAcceptorComponent } from "../components/item_acceptor";
import { enumItemProcessorTypes, ItemProcessorComponent } from "../components/item_processor";
import { Entity } from "../entity";
import { MetaBuilding } from "../meta_building";
import { GameRoot } from "../root";
import { enumHubGoalRewards } from "../tutorial_goals";
import { FormElementInput } from "../../core/modal_dialog_forms";
import { DialogWithForm } from "../../core/modal_dialog_elements";
import { T } from "../../translations";
import { MarkerComponent } from "../components/marker";

const overlayMatrix = generateMatrixRotations([1, 1, 0, 1, 1, 1, 0, 1, 1]);

export class MetaMarkerBuilding extends MetaBuilding {
    constructor() {
        super("marker");
    }

    getIsRotateable() {
        return false;
    }

    getSilhouetteColor() {
        return "#ed1d5d";
    }

    getDimensions() {
        return new Vector(1, 1);
    }

    getSpecialOverlayRenderMatrix(rotation) {
        return overlayMatrix[rotation];
    }

    /**
     * @param {GameRoot} root
     */
    getIsUnlocked(root) {
        return root.hubGoals.isRewardUnlocked(enumHubGoalRewards.reward_marker);
    }

    /**
     * Creates the entity at the given location
     * @param {Entity} entity
     */
    setupEntityComponents(entity) {
        entity.addComponent(new MarkerComponent());
    }

    /**
     * @param {Entity} entity
     */
    doPlace(entity) {
        entity.components.Marker.openSettings(true, entity);
    }
}
