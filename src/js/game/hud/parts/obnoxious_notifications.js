import { ClickDetector } from "../../../core/click_detector";
import { globalConfig } from "../../../core/config";
import { arrayDeleteValue, formatBigNumber, makeDiv, makeButton } from "../../../core/utils";
import { T } from "../../../translations";
import { enumAnalyticsDataSource } from "../../production_analytics";
import { ShapeDefinition } from "../../shape_definition";
import { enumHubGoalRewards } from "../../tutorial_goals";
import { BaseHUDPart } from "../base_hud_part";
import { Signal } from "../../../core/signal";

/**
 * Manages the pinned shapes on the left side of the screen
 */
export class HUDObnoxiousNotifications extends BaseHUDPart {
    constructor(root) {
        super(root);
        /**
         * Store a list of notifications
         * @type {Array<string>}
         */
        this.notifications = [];
        this.notificationIcons = [];

        /**
         * @type {TypedSignal<[string]>}
         */
        this.signal = new Signal("obnoxious notifications");

        /**
         * Store handles to the currently rendered elements, so we can update them more
         * convenient. Also allows for cleaning up handles.
         * @type {Array<{
         *  notification: HTMLElement
         * }>}
         */
        this.handles = [];
    }

    createElements(parent) {
        this.element = makeDiv(parent, "ingame_HUD_ObnoxiousNotifications", []);
    }

    /**
     * Serializes the notifications
     */
    serialize() {
        return {
            notifications: this.notifications,
            notificationIcons: this.notificationIcons,
        };
    }

    /**
     * Deserializes the notifications
     * @param {{ notifications: Array<string>, notificationIcons: Array<string> }} data
     */
    deserialize(data) {
        if (
            !data ||
            !data.notifications ||
            !Array.isArray(data.notifications) ||
            !data.notificationIcons ||
            !Array.isArray(data.notificationIcons)
        ) {
            return "Invalid obnoxious notifications data";
        }
        this.notifications = data.notifications;
        this.notificationIcons = data.notificationIcons;
        this.rerenderFull();
    }

    /**
     * Initializes the hud component
     */
    initialize() {
        makeDiv(this.element, null, ["title"], "Notifications");
        this.members = makeDiv(this.element, null, ["members"]);
        let minimizeButton = makeButton(this.element, ["minimize"], "...");
        this.trackClicks(minimizeButton, () => {
            if (document.documentElement.getAttribute("notifications-obnoxious") == "no") {
                document.documentElement.setAttribute("notifications-obnoxious", "yes");
                minimizeButton.innerText = "^^ Minimize";
                this.signal.dispatch("opened");
            } else {
                document.documentElement.setAttribute("notifications-obnoxious", "no");
                minimizeButton.innerText = "(open notifications)";
                this.signal.dispatch("closed");
            }
        });
        this.minimizeButton = minimizeButton;

        // Perform initial render
        this.rerenderFull();

        this.root.signals.storyGoalCompleted.add(this.levelCompletedNotification, this);
        this.root.signals.upgradePurchased.add(this.upgradePurchasedNotification, this);
        return;
        // Connect to any relevant signals
        //this.root.signals.storyGoalCompleted.add(this.rerenderFull, this);
        //this.root.signals.upgradePurchased.add(this.updateShapesAfterUpgrade, this);
        //this.root.signals.postLoadHook.add(this.rerenderFull, this);
        //this.root.hud.signals.shapePinRequested.add(this.pinNewShape, this);
        //this.root.hud.signals.shapeUnpinRequested.add(this.unpinShape, this);
    }

    levelCompletedNotification(level, reward) {
        // I could do translation here but it seems unnecessary for the purpose of the thesis

        const definition = this.root.hubGoals.computeGoalForLevel(level).definition;
        this.addNotification(definition.getHash(), "Completed level " + level + "!");
    }
    upgradePurchasedNotification(upgradeId) {
        this.addNotification("upgrade", "Purchased upgrade " + T.shopUpgrades[upgradeId].name + "!");
    }
    addNotification(icon, text) {
        this.notifications.push(text);
        this.notificationIcons.push(icon);
        if (this.notifications.length > 5) {
            this.notifications.splice(0, 1);
            this.notificationIcons.splice(0, 1);
        }
        if (document.documentElement.getAttribute("notifications-obnoxious") == "(first)") {
            document.documentElement.setAttribute("notifications-obnoxious", "yes");
            this.minimizeButton.innerText = "^^ Minimize";
            this.signal.dispatch("first notification");
        }
        this.rerenderFull();
    }

    /**
     * Rerenders the notifications
     */
    rerenderFull() {
        for (let i = 0; i < this.handles.length; ++i) {
            this.handles[i].notification.remove();
        }
        this.handles = [];

        for (let i = this.notifications.length; i-- > 0; ) {
            let notification = makeDiv(this.members, null, ["notification"]);
            let iconContainer = makeDiv(notification, null, ["icon"]);
            let icon = this.notificationIcons[i];
            if (icon != null) {
                if (icon == "upgrade") {
                    iconContainer.setAttribute("show-icon", "upgrade");
                } else {
                    const canvas = ShapeDefinition.fromShortKey(icon).generateAsCanvas(120);
                    iconContainer.appendChild(canvas);
                }
            }
            makeDiv(notification, null, ["title"], this.notifications[i]);
            this.handles.push({
                notification: notification,
            });
        }
    }

    update() {
        return;
    }
}
